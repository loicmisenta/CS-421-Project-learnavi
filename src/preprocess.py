from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STAGE1_BEHAVIORAL_FEATURES = [
    "total_events_0_2",
    "n_active_days_0_2",
    "n_click_events_0_2",
    "n_view_events_0_2",
    "n_sessions_0_2",
    "n_topics_event_0_2",
    "mean_hour_0_2",
    "std_hour_0_2",
    "total_transactions_0_2",
    "correct_rate_0_2",
    "partial_rate_0_2",
    "mean_evaluation_score_0_2",
    "avg_response_time_0_2",
    "n_documents_0_2",
    "n_topics_transaction_0_2",
    "std_evaluation_score_0_2",
    "std_response_time_0_2",
    "session_duration_mean_0_2",
    "session_duration_std_0_2",
    "time_between_sessions_mean_0_2",
    "retry_ratio_0_2",
    "review_rate_0_2",
]

STAGE2_WEEKLY_FEATURES = [
    "n_events",
    "n_active_days",
    "mean_hour",
    "n_click_events",
    "n_view_events",
    "n_sessions",
    "n_topics_event",
    "n_transactions",
    "correct_rate",
    "partial_rate",
    "mean_evaluation_score",
    "avg_response_time",
    "n_documents",
    "n_topics_transaction",
    "activity_score",
    "year",
    "day",
]


@dataclass
class CleanData:
    users: pd.DataFrame
    events: pd.DataFrame
    transactions: pd.DataFrame


@dataclass
class DatasetResult:
    df: pd.DataFrame
    feature_cols: list[str]


@dataclass
class Stage2Result:
    df_full: pd.DataFrame
    feature_cols: list[str]
    users: pd.DataFrame
    user_features: pd.DataFrame


def load_clean_data(data_dir: str | Path) -> CleanData:
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "users.csv.gz")
    events = pd.read_csv(data_dir / "events.csv.gz")
    transactions = pd.read_csv(data_dir / "transactions.csv.gz")

    users = clean_users(users)
    events = clean_events(events)
    transactions = clean_transactions(transactions)
    events, transactions = add_relative_weeks(events, transactions)

    return CleanData(users=users, events=events, transactions=transactions)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["gender"] = users["gender"].replace({"*": np.nan, "STAR": "Other"})

    for col in ["gender", "canton", "class_level", "class_id"]:
        users[col] = users[col].fillna("Unknown")

    users["study"] = (
        users["study"]
        .replace({"True": True, "False": False})
        .fillna(False)
        .astype(bool)
    )
    return users


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["user_id", "event_date"])
    return add_time_columns(events, "event_date")


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["start_time"] = pd.to_datetime(
        transactions["start_time"], errors="coerce"
    )
    transactions["commit_time"] = pd.to_datetime(
        transactions["commit_time"], errors="coerce"
    )
    transactions = transactions.dropna(subset=["user_id", "start_time"])
    transactions = add_time_columns(transactions, "start_time")

    transactions["evaluation"] = transactions["evaluation"].fillna("UNKNOWN")
    transactions["evaluation_score"] = (
        transactions["evaluation"]
        .map({"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0, "UNKNOWN": 0.0})
        .fillna(0.0)
    )
    transactions["is_correct"] = (
        transactions["evaluation"] == "CORRECT"
    ).astype(int)
    transactions["is_partial"] = (
        transactions["evaluation"] == "PARTIAL"
    ).astype(int)
    transactions["is_unknown_eval"] = (
        transactions["evaluation"] == "UNKNOWN"
    ).astype(int)
    transactions["response_time_sec"] = (
        transactions["commit_time"] - transactions["start_time"]
    ).dt.total_seconds()
    transactions.loc[
        (transactions["response_time_sec"] < 0)
        | (transactions["response_time_sec"] > 3600),
        "response_time_sec",
    ] = np.nan

    return transactions


def add_time_columns(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    timestamp = df[timestamp_col]
    df["date"] = timestamp.dt.date
    df["year"] = timestamp.dt.year
    df["month"] = timestamp.dt.month
    df["dayofweek"] = timestamp.dt.dayofweek
    df["hour"] = timestamp.dt.hour
    df["week"] = timestamp.dt.to_period("W").astype(str)
    return df


def add_relative_weeks(
    events: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_activity = (
        transactions.groupby("user_id", as_index=False)["start_time"]
        .min()
        .rename(columns={"start_time": "first_activity_time"})
    )

    events = events.merge(first_activity, on="user_id", how="left")
    transactions = transactions.merge(first_activity, on="user_id", how="left")

    events["relative_week"] = (
        (events["event_date"] - events["first_activity_time"]).dt.days // 7
    ).astype("Int64")
    transactions["relative_week"] = (
        (transactions["start_time"] - transactions["first_activity_time"]).dt.days
        // 7
    ).astype("Int64")

    return events, transactions


def build_stage1_dataset(
    data_dir: str | Path,
    save_path: str | Path | None = None,
    include_demographics: bool = True,
) -> DatasetResult:
    data = load_clean_data(data_dir)

    early_events = data.events[data.events["relative_week"].between(0, 2)].copy()
    early_transactions = data.transactions[
        data.transactions["relative_week"].between(0, 2)
    ].copy()
    returning_users = set(
        data.events.loc[data.events["relative_week"] >= 3, "user_id"].unique()
    )

    behavioural = build_stage1_behavioral_features(early_events, early_transactions)
    feature_cols = list(STAGE1_BEHAVIORAL_FEATURES)
    df = behavioural

    if include_demographics:
        demographics = build_demographic_features(data.users)
        demo_cols = [c for c in demographics.columns if c != "user_id"]
        df = df.merge(demographics, on="user_id", how="left")
        df[demo_cols] = df[demo_cols].fillna(0)
        feature_cols.extend(demo_cols)

    df["came_back"] = df["user_id"].isin(returning_users).astype(int)
    df = df[["user_id", *feature_cols, "came_back"]]

    if save_path is not None:
        save_dataframe(df, save_path)

    return DatasetResult(df=df, feature_cols=feature_cols)


def build_stage1_behavioral_features(
    events: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    parts = [
        aggregate_stage1_events(events),
        aggregate_stage1_transactions(transactions),
        aggregate_stage1_sessions(events),
        aggregate_stage1_review_rate(events),
    ]

    features = parts[0]
    for part in parts[1:]:
        features = features.merge(part, on="user_id", how="outer")

    features = features[features["total_events_0_2"].notna()].copy()
    return fill_missing_columns(features, STAGE1_BEHAVIORAL_FEATURES)


def aggregate_stage1_events(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("user_id")
        .agg(
            total_events_0_2=("event_id", "count"),
            n_active_days_0_2=("date", "nunique"),
            n_click_events_0_2=(
                "event_type",
                lambda x: (x.astype(str) == "CLICK").sum(),
            ),
            n_view_events_0_2=(
                "event_type",
                lambda x: (x.astype(str) == "VIEW").sum(),
            ),
            n_sessions_0_2=("session_id", "nunique"),
            n_topics_event_0_2=("topic_id", "nunique"),
            mean_hour_0_2=("hour", "mean"),
            std_hour_0_2=("hour", "std"),
        )
        .reset_index()
    )


def aggregate_stage1_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    features = (
        transactions.groupby("user_id")
        .agg(
            total_transactions_0_2=("transaction_id", "count"),
            correct_rate_0_2=("is_correct", "mean"),
            partial_rate_0_2=("is_partial", "mean"),
            mean_evaluation_score_0_2=("evaluation_score", "mean"),
            std_evaluation_score_0_2=("evaluation_score", "std"),
            avg_response_time_0_2=("response_time_sec", "mean"),
            std_response_time_0_2=("response_time_sec", "std"),
            n_documents_0_2=("document_id", "nunique"),
            n_topics_transaction_0_2=("topic_id", "nunique"),
        )
        .reset_index()
    )
    features["retry_ratio_0_2"] = (
        features["total_transactions_0_2"]
        / features["n_documents_0_2"].replace(0, np.nan)
    )
    return features


def aggregate_stage1_sessions(events: pd.DataFrame) -> pd.DataFrame:
    session_events = events.dropna(subset=["session_id"])
    sessions = (
        session_events.groupby(["user_id", "session_id"])
        .agg(start=("event_date", "min"), end=("event_date", "max"))
        .reset_index()
    )
    sessions["duration_sec"] = (
        sessions["end"] - sessions["start"]
    ).dt.total_seconds()

    durations = (
        sessions.groupby("user_id")
        .agg(
            session_duration_mean_0_2=("duration_sec", "mean"),
            session_duration_std_0_2=("duration_sec", "std"),
        )
        .reset_index()
    )

    sessions = sessions.sort_values(["user_id", "start"])
    sessions["prev_end"] = sessions.groupby("user_id")["end"].shift(1)
    sessions["gap_sec"] = (
        sessions["start"] - sessions["prev_end"]
    ).dt.total_seconds().clip(lower=0)
    gaps = (
        sessions.groupby("user_id")
        .agg(time_between_sessions_mean_0_2=("gap_sec", "mean"))
        .reset_index()
    )

    return durations.merge(gaps, on="user_id", how="outer")


def aggregate_stage1_review_rate(events: pd.DataFrame) -> pd.DataFrame:
    counts = events.groupby(["user_id", "action"]).size().unstack(fill_value=0)
    counts = counts.reset_index()

    for col in ["REVIEW_TASK", "SUBMIT_ANSWER"]:
        if col not in counts:
            counts[col] = 0

    counts["review_rate_0_2"] = (
        counts["REVIEW_TASK"] / counts["SUBMIT_ANSWER"].replace(0, np.nan)
    )
    return counts[["user_id", "review_rate_0_2"]]


def build_demographic_features(users: pd.DataFrame) -> pd.DataFrame:
    features = users[["user_id", "gender", "canton", "class_level", "study"]].copy()
    features["school_type"] = (
        features["class_level"].astype(str).str.split(" - ").str[0]
    )
    class_year = features["class_level"].astype(str).str.extract(r"(\d+)\. Jahr")[0]
    features["class_year"] = (
        pd.to_numeric(class_year, errors="coerce").fillna(0).astype(int)
    )
    features["study"] = features["study"].astype(int)

    categorical = pd.get_dummies(
        features[["gender", "canton", "school_type"]],
        prefix=["gender", "canton", "school"],
    ).astype(int)

    return pd.concat(
        [
            features[["user_id", "study", "class_year"]].reset_index(drop=True),
            categorical.reset_index(drop=True),
        ],
        axis=1,
    )


def build_stage2_dataset(
    data_dir: str | Path,
    horizon_weeks: int = 4,
    save_path: str | Path | None = None,
) -> Stage2Result:
    data = load_clean_data(data_dir)
    user_features = build_user_summary_features(data.users, data.events, data.transactions)
    weekly = build_weekly_features(data.events, data.transactions)
    timeline = build_full_weekly_timeline(weekly)
    timeline = add_week_dates(timeline, data.transactions)
    timeline = add_dropout_labels(timeline, horizon_weeks=horizon_weeks)

    feature_cols = list(STAGE2_WEEKLY_FEATURES)
    timeline = fill_missing_columns(timeline, feature_cols)
    timeline["is_summer"] = timeline["week_start"].apply(is_summer_vacation)

    if save_path is not None:
        save_dataframe(timeline, save_path)

    return Stage2Result(
        df_full=timeline,
        feature_cols=feature_cols,
        users=data.users,
        user_features=user_features,
    )


def build_user_summary_features(
    users: pd.DataFrame, events: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    event_features = (
        events.groupby("user_id")
        .agg(
            n_events=("event_id", "count"),
            n_active_days=("date", "nunique"),
            n_active_weeks=("week", "nunique"),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
            mean_event_hour=("hour", "mean"),
        )
        .reset_index()
    )
    transaction_features = (
        transactions.groupby("user_id")
        .agg(
            n_transactions=("transaction_id", "count"),
            n_transaction_weeks=("week", "nunique"),
            n_documents=("document_id", "nunique"),
            n_topics=("topic_id", "nunique"),
            correct_rate=("is_correct", "mean"),
            partial_rate=("is_partial", "mean"),
            avg_response_time=("response_time_sec", "mean"),
            median_response_time=("response_time_sec", "median"),
            n_challenges=("challenge_id", "nunique"),
        )
        .reset_index()
    )
    action_counts = prefixed_crosstab(events, "action", "action")
    event_type_counts = prefixed_crosstab(events, "event_type", "event_type")

    features = users.merge(event_features, on="user_id", how="left")
    features = features.merge(transaction_features, on="user_id", how="left")
    features = features.merge(action_counts, on="user_id", how="left")
    features = features.merge(event_type_counts, on="user_id", how="left")

    count_cols = [
        c
        for c in features.columns
        if c.startswith("n_")
        or c.startswith("action_")
        or c.startswith("event_type_")
    ]
    features[count_cols] = features[count_cols].fillna(0)
    features[["correct_rate", "partial_rate"]] = features[
        ["correct_rate", "partial_rate"]
    ].fillna(0)
    features["has_events"] = features["first_event"].notna().astype(int)
    features["has_transactions"] = features["n_transactions"].gt(0).astype(int)
    features["activity_span_days"] = (
        features["last_event"] - features["first_event"]
    ).dt.days
    features["events_per_active_day"] = safe_divide(
        features["n_events"], features["n_active_days"]
    )
    features["events_per_active_week"] = safe_divide(
        features["n_events"], features["n_active_weeks"]
    )
    features["transactions_per_active_week"] = safe_divide(
        features["n_transactions"], features["n_transaction_weeks"]
    )
    features["has_response_time"] = features["avg_response_time"].notna().astype(int)
    return features


def build_weekly_features(events: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    event_week = (
        events.groupby(["user_id", "relative_week"])
        .agg(
            n_events=("event_id", "count"),
            n_active_days=("date", "nunique"),
            mean_hour=("hour", "mean"),
            n_click_events=("event_type", lambda x: (x.astype(str) == "CLICK").sum()),
            n_view_events=("event_type", lambda x: (x.astype(str) == "VIEW").sum()),
            n_sessions=("session_id", "nunique"),
            n_topics_event=("topic_id", "nunique"),
        )
        .reset_index()
    )
    transaction_week = (
        transactions.groupby(["user_id", "relative_week"])
        .agg(
            n_transactions=("transaction_id", "count"),
            correct_rate=("is_correct", "mean"),
            partial_rate=("is_partial", "mean"),
            mean_evaluation_score=("evaluation_score", "mean"),
            avg_response_time=("response_time_sec", "mean"),
            n_documents=("document_id", "nunique"),
            n_topics_transaction=("topic_id", "nunique"),
        )
        .reset_index()
    )

    weekly = event_week.merge(
        transaction_week, on=["user_id", "relative_week"], how="outer"
    )
    count_cols = [
        "n_events",
        "n_active_days",
        "n_click_events",
        "n_view_events",
        "n_sessions",
        "n_topics_event",
        "n_transactions",
        "n_documents",
        "n_topics_transaction",
    ]
    weekly[count_cols] = weekly[count_cols].fillna(0)
    weekly["activity_score"] = (
        weekly["n_events"]
        + weekly["n_transactions"]
        + weekly["n_click_events"]
        + weekly["n_view_events"]
    )
    return weekly


def build_full_weekly_timeline(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, user_df in weekly.groupby("user_id"):
        max_week = user_df["relative_week"].max()
        if pd.isna(max_week):
            continue
        full_weeks = pd.DataFrame(
            {
                "user_id": user_id,
                "relative_week": np.arange(0, int(max_week) + 1),
            }
        )
        rows.append(
            full_weeks.merge(user_df, on=["user_id", "relative_week"], how="left")
        )
    return pd.concat(rows, ignore_index=True)


def add_week_dates(timeline: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    first_activity = (
        transactions.groupby("user_id", as_index=False)["first_activity_time"]
        .first()
        .rename(columns={"first_activity_time": "timeline_start"})
    )
    timeline = timeline.merge(first_activity, on="user_id", how="left")
    timeline["week_start"] = timeline["timeline_start"] + pd.to_timedelta(
        timeline["relative_week"] * 7, unit="D"
    )
    timeline["year"] = timeline["week_start"].dt.year
    timeline["day"] = timeline["week_start"].dt.dayofyear
    return timeline.drop(columns=["timeline_start"])


def add_dropout_labels(
    timeline: pd.DataFrame, horizon_weeks: int = 4
) -> pd.DataFrame:
    timeline = timeline.sort_values(["user_id", "relative_week"]).copy()
    timeline["is_active"] = timeline["n_events"].fillna(0).gt(0).astype(int)

    labels = []
    for _, user_df in timeline.groupby("user_id", sort=False):
        activity = user_df["is_active"].to_numpy()
        week_start = user_df["week_start"].to_numpy()
        user_labels = np.full(len(user_df), np.nan)

        for idx in range(len(user_df)):
            if idx + horizon_weeks >= len(user_df):
                continue

            future = activity[idx + 1 : idx + 1 + horizon_weeks]
            if not np.all(future == 0):
                user_labels[idx] = 0
                continue

            gap_start = pd.Timestamp(week_start[idx]) + pd.Timedelta(days=7)
            gap_end = pd.Timestamp(week_start[idx]) + pd.Timedelta(
                days=7 * horizon_weeks
            )
            user_labels[idx] = 0 if overlaps_summer(gap_start, gap_end) else 1

        labels.extend(user_labels)

    timeline["is_dropout_point"] = labels
    return timeline


def fill_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df:
            df[col] = 0.0
    df[columns] = df[columns].fillna(0.0)
    return df


def prefixed_crosstab(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    counts = pd.crosstab(df["user_id"], df[column])
    counts.columns = [f"{prefix}_{str(c).lower()}" for c in counts.columns]
    return counts.reset_index()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    df.to_csv(path, index=False, compression=compression)


def is_summer_vacation(date: pd.Timestamp) -> bool:
    return date.month in [7, 8]


def overlaps_summer(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if pd.isna(start) or pd.isna(end):
        return False

    for year in range(start.year, end.year + 1):
        july_start = pd.Timestamp(year=year, month=7, day=1)
        august_end = pd.Timestamp(year=year, month=8, day=31)
        if not (end < july_start or start > august_end):
            return True
    return False


def preprocess_stage1(*args, **kwargs) -> DatasetResult:
    return build_stage1_dataset(*args, **kwargs)


def preprocess_stage2(*args, **kwargs) -> Stage2Result:
    return build_stage2_dataset(*args, **kwargs)
