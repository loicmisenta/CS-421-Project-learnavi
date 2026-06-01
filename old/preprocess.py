import pandas as pd
import gc
import numpy as np

def preprocess(DATA_DIR):
    users = pd.read_csv('{}/users.csv.gz'.format(DATA_DIR))
    events = pd.read_csv('{}/events.csv.gz'.format(DATA_DIR))
    transactions = pd.read_csv('{}/transactions.csv.gz'.format(DATA_DIR))
    
    users_clean = users.copy()  # Work on a copy to keep raw data unchanged

    # Normalize gender values and fill missing entries
    users_clean['gender'] = (
        users_clean['gender']
        .replace({'*': np.nan, 'STAR': 'Other'})
        .fillna('Unknown')
    )

    # Fill missing categorical profile fields
    users_clean['canton'] = users_clean['canton'].fillna('Unknown')
    users_clean['class_level'] = users_clean['class_level'].fillna('Unknown')
    users_clean['class_id'] = users_clean['class_id'].fillna('Unknown')

    # Ensure study is boolean
    users_clean['study'] = users_clean['study'].replace({'True': True, 'False': False}).fillna(False).astype(bool)
    
    del users
    gc.collect()

    events_clean = events.copy()  # Keep raw events unchanged

    events_clean['event_date'] = pd.to_datetime(events_clean['event_date'], errors='coerce')
    events_clean = events_clean.dropna(subset=['user_id', 'event_date'])  # Remove unusable rows

    # Time-based features for analysis/aggregation
    events_clean['date'] = events_clean['event_date'].dt.date
    events_clean['year'] = events_clean['event_date'].dt.year
    events_clean['month'] = events_clean['event_date'].dt.month
    events_clean['dayofweek'] = events_clean['event_date'].dt.dayofweek
    events_clean['hour'] = events_clean['event_date'].dt.hour
    events_clean['week'] = events_clean['event_date'].dt.to_period('W').astype(str)
    
    del events
    gc.collect()
    
    transactions_clean = transactions.copy()  # keep raw transactions unchanged

    # parse timestamps and drop rows missing required keys/time
    transactions_clean['start_time'] = pd.to_datetime(transactions_clean['start_time'], errors='coerce')
    transactions_clean['commit_time'] = pd.to_datetime(transactions_clean['commit_time'], errors='coerce')
    transactions_clean = transactions_clean.dropna(subset=['user_id', 'start_time'])

    # derive time features from start_time
    transactions_clean['date'] = transactions_clean['start_time'].dt.date
    transactions_clean['year'] = transactions_clean['start_time'].dt.year
    transactions_clean['month'] = transactions_clean['start_time'].dt.month
    transactions_clean['dayofweek'] = transactions_clean['start_time'].dt.dayofweek
    transactions_clean['hour'] = transactions_clean['start_time'].dt.hour
    transactions_clean['week'] = transactions_clean['start_time'].dt.to_period('W').astype(str)
    
    del transactions
    gc.collect()
    
    # Clean evaluation labels and derive outcome flags
    transactions_clean['evaluation'] = transactions_clean['evaluation'].fillna('UNKNOWN')

    # Soft correctness score
    transactions_clean['evaluation_score'] = transactions_clean['evaluation'].map({
        'CORRECT': 1.0,
        'PARTIAL': 0.5,
        'WRONG': 0.0,
        'UNKNOWN': 0.0
    }).fillna(0.0)

    transactions_clean['is_correct'] = (transactions_clean['evaluation'] == 'CORRECT').astype(int)
    transactions_clean['is_partial'] = (transactions_clean['evaluation'] == 'PARTIAL').astype(int)
    transactions_clean['is_unknown_eval'] = (transactions_clean['evaluation'] == 'UNKNOWN').astype(int)

    # Compute response time in seconds
    transactions_clean['response_time_sec'] = (
        transactions_clean['commit_time'] - transactions_clean['start_time']
    ).dt.total_seconds()

    # Remove implausible response times (discard values > 1 hour but could be adjusted)
    transactions_clean.loc[
        (transactions_clean['response_time_sec'] < 0) | (transactions_clean['response_time_sec'] > 3600),
        'response_time_sec'
    ] = np.nan
    
    # Define week 0 as each user's first recorded transaction
    first_activity = (
        transactions_clean.groupby('user_id', as_index=False)['start_time']
        .min()
        .rename(columns={'start_time': 'first_activity_time'})
    )

    events_clean = events_clean.merge(first_activity, on='user_id', how='left')
    transactions_clean = transactions_clean.merge(first_activity, on='user_id', how='left')

    events_clean['relative_week'] = (
        (events_clean['event_date'] - events_clean['first_activity_time']).dt.days // 7
    ).astype('Int64')
    transactions_clean['relative_week'] = (
        (transactions_clean['start_time'] - transactions_clean['first_activity_time']).dt.days // 7
    ).astype('Int64')
    
    # Aggregate per-user event activity features
    event_user_features = events_clean.groupby('user_id').agg(
        n_events=('event_id', 'count'),
        n_active_days=('date', 'nunique'),
        n_active_weeks=('week', 'nunique'),
        first_event=('event_date', 'min'),
        last_event=('event_date', 'max'),
        mean_event_hour=('hour', 'mean')
    ).reset_index()
    
    # Count how many times each action appears per user
    action_counts = pd.crosstab(events_clean['user_id'], events_clean['action'])

    # Prefix action columns for clarity (e.g., action_login, action_next, ...)
    action_counts.columns = [f'action_{c.lower()}' for c in action_counts.columns]

    # Bring user_id back as a regular column for merging
    action_counts = action_counts.reset_index()
    
    # Count event types per user
    event_type_counts = pd.crosstab(events_clean['user_id'], events_clean['event_type'])
    event_type_counts.columns = [f'event_type_{c.lower()}' for c in event_type_counts.columns]
    event_type_counts = event_type_counts.reset_index()
    
    # Per-user transaction behavior summary
    transaction_user_features = transactions_clean.groupby('user_id').agg(
        n_transactions=('transaction_id', 'count'),      # total attempts
        n_transaction_weeks=('week', 'nunique'),         # weeks with at least one transaction
        n_documents=('document_id', 'nunique'),          # unique documents worked on
        n_topics=('topic_id', 'nunique'),                # unique topics covered
        correct_rate=('is_correct', 'mean'),
        partial_rate=('is_partial', 'mean'),
        avg_response_time=('response_time_sec', 'mean'),
        median_response_time=('response_time_sec', 'median'),
        n_challenges=('challenge_id', 'nunique')         # distinct challenge sets
    ).reset_index()
    
    user_features = users_clean.merge(event_user_features, on='user_id', how='left')
    user_features = user_features.merge(action_counts, on='user_id', how='left')
    user_features = user_features.merge(event_type_counts, on='user_id', how='left')
    user_features = user_features.merge(transaction_user_features, on='user_id', how='left')

    # Fill count-like features with 0
    count_cols = [
        col for col in user_features.columns
        if col.startswith('n_') or col.startswith('action_') or col.startswith('event_type_')
    ]
    user_features[count_cols] = user_features[count_cols].fillna(0)

    # Fill main rates with 0 when user has no transactions 
    rate_cols = ['correct_rate', 'partial_rate']
    user_features[rate_cols] = user_features[rate_cols].fillna(0)

    # Indicators for missing activity / transactions
    user_features['has_events'] = user_features['first_event'].notna().astype(int)
    user_features['has_transactions'] = user_features['n_transactions'].gt(0).astype(int)

    # Activity span
    user_features['activity_span_days'] = (
        user_features['last_event'] - user_features['first_event']
    ).dt.days

    # Frequency features
    user_features['events_per_active_day'] = (
        user_features['n_events'] / user_features['n_active_days'].replace(0, np.nan)
    )
    user_features['events_per_active_week'] = (
        user_features['n_events'] / user_features['n_active_weeks'].replace(0, np.nan)
    )
    user_features['transactions_per_active_week'] = (
        user_features['n_transactions'] / user_features['n_transaction_weeks'].replace(0, np.nan)
    )

    # Keep response-time NaNs, but mark availability
    user_features['has_response_time'] = user_features['avg_response_time'].notna().astype(int)
    
    # Weekly per-user event activity features on the relative timeline
    events_week = events_clean.groupby(['user_id', 'relative_week']).agg(
        n_events=('event_id', 'count'),
        n_active_days=('date', 'nunique'),
        mean_hour=('hour', 'mean'),
        n_click_events=('event_type', lambda x: (x.astype(str) == 'CLICK').sum()),
        n_view_events=('event_type', lambda x: (x.astype(str) == 'VIEW').sum()),
        n_sessions=('session_id', 'nunique'),
        n_topics_event=('topic_id', 'nunique')
    ).reset_index()
    
    # Weekly transaction-level features per user on the relative timeline
    transactions_week = transactions_clean.groupby(['user_id', 'relative_week']).agg(
        n_transactions=('transaction_id', 'count'),
        correct_rate=('is_correct', 'mean'),
        partial_rate=('is_partial', 'mean'),
        mean_evaluation_score=('evaluation_score', 'mean'),
        avg_response_time=('response_time_sec', 'mean'),
        n_documents=('document_id', 'nunique'),
        n_topics_transaction=('topic_id', 'nunique')
    ).reset_index()
    
    # Combine weekly event and transaction features per user
    user_week_features = events_week.merge(transactions_week, on=['user_id', 'relative_week'], how='outer')

    weekly_count_cols = [
        'n_events', 'n_active_days', 'n_click_events', 'n_view_events',
        'n_sessions', 'n_topics_event', 'n_transactions', 'n_documents',
        'n_topics_transaction'
    ]
    user_week_features[weekly_count_cols] = user_week_features[weekly_count_cols].fillna(0)

    # Simple overall weekly engagement score
    user_week_features['activity_score'] = (
        user_week_features['n_events']
        + user_week_features['n_transactions']
        + user_week_features['n_click_events']
        + user_week_features['n_view_events']
    )
    
    df_dropout, week_dropout_df = compute_dropout(user_week_features, user_features)
    
    df_full = build_full_timeline(user_week_features)
    
    feature_cols = [
        "n_events", "n_active_days", "mean_hour",
        "n_click_events", "n_view_events", "n_sessions",
        "n_topics_event", "n_transactions",
        "correct_rate", "partial_rate", "mean_evaluation_score",
        "avg_response_time", "n_topics_transaction","n_documents",
        "activity_score"
    ]

    df_full[feature_cols] = df_full[feature_cols].fillna(0)
    
    df_full = compute_dropout_full_timeline(df_full, user_features)
    
    # get the starting week (relative_week = 0) for each user
    start_dates = (
        week_dropout_df[week_dropout_df["relative_week"] == 0]
        [["user_id", "week_start"]]
        .rename(columns={"week_start": "start_date"})
    )

    # merge with df_full
    df_full = df_full.merge(start_dates, on="user_id", how="left")

    # reconstruct the actual date for each row
    df_full["date"] = df_full["start_date"] + pd.to_timedelta(df_full["relative_week"] * 7, unit="D")

    # extract year and day of year
    df_full["year"] = df_full["date"].dt.year
    df_full["day"] = df_full["date"].dt.dayofyear

    # drop helper columns if you don't need them
    df_full = df_full.drop(columns=["start_date", "date"])

    feature_cols.append("year")
    feature_cols.append("day")
    
    df_full["is_summer"] = df_full["week_start"].apply(is_summer_vacation)
    
    return users_clean, events_clean, transactions_clean, event_user_features, feature_cols, df_full
    
    
    



def overlaps_summer(row):
        if pd.isna(row["gap_start"]) or pd.isna(row["gap_end"]):
            return False

        start = row["gap_start"]
        end = row["gap_end"]

        # iterate over years spanned by the gap
        for year in range(start.year, end.year + 1):
            july_start = pd.Timestamp(year=year, month=7, day=1)
            aug_end = pd.Timestamp(year=year, month=8, day=31)

            # overlap condition
            if not (end < july_start or start > aug_end):
                return True

        return False

def compute_dropout(events_week, users_features):

    df = events_week.merge(
        users_features[["user_id", "first_event"]],
        on="user_id",
        how="left"
    )
    # compute week start date
    df["week_start"] = df["first_event"] + pd.to_timedelta(df["relative_week"] * 7, unit="D")

    # ensure ordering
    df = df.sort_values(["user_id", "relative_week"]).copy()

    # gap between weeks
    df["week_diff"] = df.groupby("user_id")["relative_week"].diff()

    df["prev_week_start"] = df.groupby("user_id")["week_start"].shift(1)

    df["gap_start"] = df["prev_week_start"] + pd.Timedelta(days=7)
    df["gap_end"] = df["week_start"] - pd.Timedelta(days=7)

    df["overlaps_summer"] = df.apply(overlaps_summer, axis=1)

    # dropout condition
    df["is_dropout_point"] = (
        (df["week_diff"] > 4) &            # >= 4 weeks gap
        (~df["overlaps_summer"])           # gap not during summer
    )
    # shift dropout to previous week (within each user)
    df["is_dropout_point"] = df.groupby("user_id")["is_dropout_point"].shift(-1).fillna(False)

    today = pd.Timestamp.today()

    # identify last week per user
    df["is_last_week"] = df.groupby("user_id")["relative_week"].transform("max") == df["relative_week"]

    # compute how old the last week is
    df["weeks_since"] = (today - df["week_start"]).dt.days / 7

    # apply rule only on last week
    df.loc[df["is_last_week"], "is_dropout_point"] = df.loc[df["is_last_week"], "weeks_since"] > 4

    # first dropout
    dropout_df = df[df["is_dropout_point"]].copy()
    dropout_df["dropout_week"] = dropout_df["relative_week"].shift(1)

    first_dropout = (
        dropout_df.groupby("user_id")
        .first()[["dropout_week"]]
        .reset_index()
    )
    first_dropout["dropout_indicator"] = 1

    # users without dropout -> last week
    last_weeks = (
        df.groupby("user_id")["relative_week"]
        .max()
        .reset_index()
        .rename(columns={"relative_week": "dropout_week"})
    )
    last_weeks["dropout_indicator"] = 0

    # final merge
    result = last_weeks.merge(first_dropout, on="user_id", how="left", suffixes=("", "_drop"))

    # if dropout exists, we overwrite
    result["dropout_indicator"] = result["dropout_indicator_drop"].fillna(result["dropout_indicator"])
    result["dropout_week"] = result["dropout_week_drop"].fillna(result["dropout_week"])

    # cleanup
    result = result[["user_id", "dropout_indicator", "dropout_week"]]

    return result, df
    
    

def build_full_timeline(df):
    users = df["user_id"].unique()
    full_data = []

    for user in users:
        user_df = df[df["user_id"] == user]
        max_week = user_df["relative_week"].max()

        full_weeks = pd.DataFrame({
            "user_id": user,
            "relative_week": np.arange(0, max_week + 1)
        })

        merged = full_weeks.merge(
            user_df,
            on=["user_id", "relative_week"],
            how="left"
        )

        full_data.append(merged)

    return pd.concat(full_data, ignore_index=True)
    
    


def overlaps_summer_period(start, end):
    if pd.isna(start) or pd.isna(end):
        return False

    for year in range(start.year, end.year + 1):
        july_start = pd.Timestamp(year=year, month=7, day=1)
        aug_end = pd.Timestamp(year=year, month=8, day=31)

        if not (end < july_start or start > aug_end):
            return True

    return False


def compute_dropout_full_timeline(df, users_features, horizon=4):
    df = df.merge(
        users_features[["user_id", "first_event"]],
        on="user_id",
        how="left"
    )

    # compute week_start
    df["week_start"] = df["first_event"] + pd.to_timedelta(df["relative_week"] * 7, unit="D")

    df = df.sort_values(["user_id", "relative_week"]).copy()

    # activity indicator
    df["is_active"] = (df["n_events"] > 0).astype(int)

    labels = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("relative_week")
        activity = user_df["is_active"].values
        weeks = user_df["week_start"].values
        n = len(user_df)

        user_labels = np.full(n, np.nan)

        for i in range(n):
            # need full horizon
            if i + horizon < n:
                future_activity = activity[i+1:i+1+horizon]

                if np.all(future_activity == 0):
                    gap_start = weeks[i] + np.timedelta64(7, 'D')
                    gap_end = weeks[i] + np.timedelta64(7*horizon, 'D')

                    if not overlaps_summer_period(pd.Timestamp(gap_start), pd.Timestamp(gap_end)):
                        user_labels[i] = 1
                    else:
                        user_labels[i] = 0
                else:
                    user_labels[i] = 0
            else:
                user_labels[i] = np.nan  # cannot determine

        labels.extend(user_labels)

    df["is_dropout_point"] = labels
    return df

def is_summer_vacation(date):
    return (date.month == 7) or (date.month == 8)