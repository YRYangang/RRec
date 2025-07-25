import gzip
import html
import json
import os
import re
from typing import Literal

import fire
import pandas as pd
import requests
from datasets import DatasetDict, Dataset
from tqdm import tqdm

JUMP_OFFSET = pd.DateOffset(months=3)
MAX_LEN = 20

def gao(metadata, reviews: pd.DataFrame, K, start_time, end_time):
    print(f"from {start_time} to {end_time}")
    current_reviews = reviews[(reviews["time"] >= start_time) & (reviews["time"] <= end_time)].copy()

    num_users = len(current_reviews["user_id"].unique())
    num_items = len(current_reviews["parent_asin"].unique())
    num_reviews = len(current_reviews)

    while True:
        # filter with remove_users and remove_items
        user_df = current_reviews.groupby("user_id").size().reset_index(name="count")
        item_df = current_reviews.groupby("parent_asin").size().reset_index(name="count")
        users2remove = user_df[user_df["count"] < K]["user_id"].tolist()
        items2remove = item_df[item_df["count"] < K]["parent_asin"].tolist()
        current_reviews = current_reviews[
            ~(
                (current_reviews["user_id"].isin(users2remove))
                | (current_reviews["parent_asin"].isin(items2remove))
            )
        ]

        num_reviews = len(current_reviews)
        num_users = len(current_reviews["user_id"].unique())
        num_items = len(current_reviews["parent_asin"].unique())

        if len(users2remove) == 0 and len(items2remove) == 0:
            break

    if num_items < 10000:
        start_datetime = pd.to_datetime(start_time)
        if start_datetime.year > reviews["time"].min().year:
            start_time = pd.to_datetime(start_time) - JUMP_OFFSET
            start_time = start_time.strftime("%Y-%m-%d")

            print(
                "[After filtering] " f"users: {num_users}, " f"items: {num_items}, " f"reviews: {num_reviews}"
            )
            print("Not enough items, try to get more items")
            return gao(
                metadata=metadata,
                reviews=reviews,
                K=K,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            print("Not enough items, but already reached the minimum year")

    print("Data filtering done!")
    print(
        "[Final Stats] "
        f"users: {num_users}, "
        f"items: {num_items}, "
        f"reviews: {num_reviews}, "
        f"density: {num_reviews / (num_users * num_items)}"
    )

    return current_reviews, metadata


def save_data(
    current_reviews,
    metadata,
    file_name,
    data_root_dir,
    window_size=10,
    index_item_with_pad=True,
    add_interaction_id=True,
):
    new_reviews = current_reviews.to_dict(orient="records")
    items = current_reviews["parent_asin"].unique().tolist()
    asin2title = {
        item["parent_asin"]: item["title"]
        for item in tqdm(metadata, desc="Creating asin2title mapping")
        if item["parent_asin"] in items
    }
    new_items = set()
    if index_item_with_pad:
        asin2title["pad_asin"] = "pad_title"
        asin2id = {asin: idx + 1 for idx, asin in enumerate(asin2title.keys())}
        asin2id["pad_asin"] = 0
        new_items.add("pad_asin")
    else:
        asin2id = {item: idx for idx, item in enumerate(asin2title.keys())}

    interact = {}

    for review in new_reviews:
        user = review["user_id"]
        item = review["parent_asin"]
        if user not in interact:
            interact[user] = {
                "items": [],
                "ratings": [],
                "timestamps": [],
            }
        new_items.add(item)
        interact[user]["items"].append(item)
        interact[user]["ratings"].append(review["rating"])
        interact[user]["timestamps"].append(review["timestamp"])

    interaction_list = []
    for key in interact.keys():
        items = interact[key]["items"]
        ratings = interact[key]["ratings"]
        timestamps = interact[key]["timestamps"]

        all = list(zip(items, ratings, timestamps))
        res = sorted(all, key=lambda x: int(x[2]))
        items, ratings, timestamps = zip(*res)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)

        interact[key]["items"] = items
        interact[key]["ratings"] = ratings
        interact[key]["timestamps"] = timestamps
        interact[key]["item_ids"] = [asin2id[item] for item in items]
        interact[key]["title"] = [asin2title[item] for item in items]

        for i in range(1, len(items)):
            st = max(i - window_size, 0)
            assert i - st > 0, f"i: {i}, st: {st}"
            interaction_list.append(
                [
                    key,
                    interact[key]["items"][st:i],
                    interact[key]["items"][i],
                    interact[key]["item_ids"][st:i],
                    interact[key]["item_ids"][i],
                    interact[key]["title"][st:i],
                    interact[key]["title"][i],
                    interact[key]["ratings"][st:i],
                    interact[key]["ratings"][i],
                    interact[key]["timestamps"][st:i],
                    interact[key]["timestamps"][i],
                ]
            )
    print(f"interaction_list: {len(interaction_list)}")

    # split train val test
    interaction_list = sorted(interaction_list, key=lambda x: int(x[-1]))

    os.makedirs(data_root_dir, exist_ok=True)
    column_names = [
        "user_id",
        "item_asins",
        "item_asin",
        "history_item_id",
        "item_id",
        "history_item_title",
        "item_title",
        "history_rating",
        "rating",
        "history_timestamp",
        "timestamp",
    ]

    if add_interaction_id:
        for i in range(len(interaction_list)):
            interaction_list[i].append(i)
        column_names.append("interaction_id")
    # Create a DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(
                pd.DataFrame(interaction_list[: int(len(interaction_list) * 0.8)], columns=column_names)
            ),
            "valid": Dataset.from_pandas(
                pd.DataFrame(
                    interaction_list[int(len(interaction_list) * 0.8) : int(len(interaction_list) * 0.9)],
                    columns=column_names,
                )
            ),
            "test": Dataset.from_pandas(
                pd.DataFrame(interaction_list[int(len(interaction_list) * 0.9) :], columns=column_names)
            ),
            "item_info": Dataset.from_pandas(
                item_info_to_df(metadata, new_items, asin2id, add_pad_item=index_item_with_pad)
            ),
        }
    )
    dataset_dir = os.path.join(data_root_dir, file_name)
    dataset_dict.save_to_disk(dataset_dir)

    print(
        f"Train: {len(dataset_dict['train'])}, "
        f"Val: {len(dataset_dict['valid'])}, "
        f"Test: {len(dataset_dict['test'])}, "
        f"Items: {len(dataset_dict['item_info'])}"
    )


def item_info_to_df(metadata, new_items, asin2id, add_pad_item=True):
    """
    ['main_category', 'title', 'subtitle', 'average_rating', 'rating_number',
       'features', 'description', 'price', 'store', 'categories', 'details',
       'parent_asin']

    """
    metadata_df = pd.DataFrame(metadata)
    metadata_df.drop(["videos", "author", "bought_together", "images"], axis=1, inplace=True)
    # filter the items according to the new_items_asins
    metadata_df = metadata_df[metadata_df["parent_asin"].isin(new_items)]
    metadata_df.loc[:, "item_id"] = metadata_df["parent_asin"].map(asin2id)

    metadata_df = metadata_df.astype({"rating_number": "int"})
    metadata_df["details"] = metadata_df["details"].apply(json.dumps)

    metadata_df.loc[:, "subtitle"] = metadata_df["subtitle"].apply(
        lambda x: "" if x is None or str(x).lower() == "nan" or str(x).lower() == "none" else str(x)
    )

    metadata_df.loc[:, "description"] = metadata_df["description"].apply(
        lambda x: list(set(x)) if isinstance(x, list) else [x] if isinstance(x, str) else []
    )

    metadata_df["features"] = metadata_df["features"].apply(lambda x: x if isinstance(x, list) else [])

    # add one row of pad item
    if add_pad_item:
        metadata_df.loc[len(metadata_df)] = {
            "main_category": "pad_category",
            "title": "pad_title",
            "subtitle": "pad_subtitle",
            "average_rating": 0,
            "rating_number": 0,
            "features": [],
            "description": [],
            "price": 0,
            "store": "pad_store",
            "categories": [],
            "details": json.dumps({}),
            "parent_asin": "pad_asin",
            "item_id": 0,
        }
    return metadata_df

    # desirable features


def _download_raw(path: str, type: str = "review", category: str = "Movies_and_TV", verbose=True) -> str:
    """
    Downloads the raw data file from the specified URL and saves it locally.

    Args:
        path (str): The path to the directory where the file will be saved.
        type (str, optional): The type of data to download. Defaults to 'reviews'.

    Returns:
        str: The local file path where the downloaded file is saved.
    """
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    if type == "review":
        url += f"benchmark/5core/rating_only/{category}.csv.gz"
    elif type == "meta":
        url += f"raw/meta_categories/meta_{category}.jsonl.gz"
    else:
        raise ValueError(f"Invalid type: {type}")

    base_name = os.path.basename(url)
    local_filepath = os.path.join(path, base_name)
    if not os.path.exists(local_filepath):
        _download_file_with_progress(url, local_filepath) if verbose else _download_file(url, local_filepath)
    else:
        print(f"{os.path.basename(local_filepath)} already exists. Skipping download.")
    return local_filepath


def _download_file_with_progress(url: str, path: str) -> None:
    """
    Downloads a file from a URL and saves it locally with a progress bar.

    Args:
        url (str): The URL of the file to download.
        path (str): The local path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {os.path.basename(path)} successfully.")
    else:
        raise ValueError(
            f"Failed to download {os.path.basename(path)}. HTTP status code: {response.status_code}"
        )


def _download_file(url: str, path: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        raise ValueError(
            f"Failed to download {os.path.basename(path)}. HTTP status code: {response.status_code}"
        )


def _parse_gz(path: str, desc: str):
    with gzip.open(path, "r") as g:
        for l in tqdm(g, unit="lines", desc=desc):
            yield json.loads(l.strip())


def _parse_gz_fast(path: str, data_root_dir: str):
    # 使用 tempfile 创建临时目录

    tmpdir = os.path.join(data_root_dir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    df_file_save_path = os.path.basename(path).replace(".gz", "") + ".parquet"
    if os.path.exists(os.path.join(tmpdir, df_file_save_path)):
        print(f"Found existing parquet file at {tmpdir}/{df_file_save_path}. Loading...")
        df = pd.read_parquet(os.path.join(tmpdir, df_file_save_path))
        return df

    print(f"Extracting {path} to {tmpdir}/{df_file_save_path}")
    with gzip.open(path, "r") as g:
        content = g.read().decode("utf-8")  # 将 gzip 内容解码为字符串
        df = pd.read_json(content, lines=True)  # 读取 jsonl 文件

    print(f"Extracted {len(df)} records from {path}")
    df.to_parquet(os.path.join(tmpdir, df_file_save_path), index=False)  # 保存为 parquet 文件

    file_size = os.path.getsize(os.path.join(tmpdir, df_file_save_path)) / 1024 / 1024
    print(f"Saved parquet file to {tmpdir}/{df_file_save_path}, size: {file_size:.2f} MB")

    return df


def load_items(path: str):
    items = []
    items_ids = []
    
    num_no_title = 0
    num_invalid_title = 0
    num_too_long_title = 0
    num_invalid_price = 0

    for i, item in enumerate(_parse_gz(path, "Loading items")):
        if "title" not in item or item["title"] is None:
            num_no_title += 1
            continue
        if item["title"].find("<span id") > -1:
            num_invalid_title += 1
            continue
        item["title"] = item["title"].replace("&quot;", '"').replace("&amp;", "&").strip(" ").strip('"')

        if len(item["title"].split(" ")) > MAX_LEN:
            num_too_long_title += 1
            continue
        if len(item["title"]) <= 1:  # remove the item without title
            num_no_title += 1
            continue
        price = item.get("price", None)
        try:
            price = float(price)
        except:
            num_invalid_price += 1
            continue
        for feature_name, feature in item.items():
            if isinstance(feature, str):
                sentence = clean_text(feature)
                item[feature_name] = sentence
        items.append(item)
        items_ids.append(item["parent_asin"])

    print(f"Loaded {len(items)} items, {i - len(items)} items removed")
    print(
        f"num_no_title: {num_no_title}, "
        f"num_invalid_title: {num_invalid_title}, "
        f"num_too_long_title: {num_too_long_title}, "
        f"num_invalid_price: {num_invalid_price}"
    )

    return items, items_ids


def clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text


def main(
    category: str = "CDs_and_Vinyl",
    K: int = 0,
    st_year: int = 2022,
    st_month: int = 10,
    ed_year: int = 2023,
    ed_month: int = 10,
    window_size: int = 20,
    output: bool = True,
    data_root_dir="data",
    postfix="",
):
    review_path = _download_raw(data_root_dir, type="review", category=category)
    meta_path = _download_raw(data_root_dir, type="meta", category=category)

    items, items_ids = load_items(meta_path)

    df = pd.read_csv(os.path.join(data_root_dir, f"{category}.csv.gz"), encoding="utf-8")

    # filter the reviews according to the items
    df = df[df["parent_asin"].isin(items_ids)]
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")

    print(f"from {category} items: {len(items)} reviews: {len(df)}")

    new_reviews, items = gao(
        items, df, K=K, start_time=f"{st_year}-{st_month}-01", end_time=f"{ed_year}-{ed_month}-01"
    )

    file_name = f"{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}"
    if postfix:
        file_name += f"_{postfix}"

    if output:
        save_data(new_reviews, items, file_name, data_root_dir, window_size=window_size)


if __name__ == "__main__":
    fire.Fire(main)
