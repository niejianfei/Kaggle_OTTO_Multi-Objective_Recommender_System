import pandas as pd
from ranker import generate_submission
from ranker import user_sample


def submission(candidate_type):
    sub = pd.DataFrame()
    for t in candidate_type:
        df = pd.read_parquet(f'/home/niejianfei/otto/LB/submission/sub_{t}.pqt')
        df = df.loc[df.session_type.str.contains(t)]
        sub = sub.append(df)
    return sub


if __name__ == '__main__':
    candidate_type = ['clicks', 'carts', 'orders']
    generate_submission('test', 'LB', candidate_type, user_sample(0.5), 'final_all_data')

    submission_final = submission(candidate_type)
    submission_final.to_csv(f'/home/niejianfei/otto/LB/submission/submission_final.csv', index=False)
