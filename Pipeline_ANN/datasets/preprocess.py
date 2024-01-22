import pandas as pd
from .encoder import Encoder_Module
from .external import external_data

class preprosess_Module:
    def __init__(self,df:pd.DataFrame):
        self.df = df
    
    def preprocess(self, df:pd.DataFrame, df_tst:pd.DataFrame, features:iter=None):
        df.dropna(axis=0, subset=['ECLO'], inplace=True)
        
        df_num = df.select_dtypes(exclude=['object'])
        df_tst_num = df_tst.select_dtypes(exclude=['object'])
        
        missing_df = (df_num.isnull().sum())
        
        all_columns = df_num.columns
        df_num = df_num.drop(all_columns[missing_df > 0], axis=1)
        df_tst_num = df_tst_num.drop(all_columns[missing_df > 0], axis=1)
        
        df_num.fillna(df_num.min(), inplace=True)
        df_tst_num.fillna(df_tst_num.min(), inplace=True)
        
        df_cat = df.select_dtypes(include=['object'])
        df_cat_tst = df_tst.select_dtypes(include=['object'])
        
        time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})'

        df_cat[['연', '월', '일', '시간']] = df_cat['사고일시'].str.extract(time_pattern)
        df_cat[['연', '월', '일', '시간']] = df_cat[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다 
        df_cat = df_cat.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다 

        df_cat_tst[['연', '월', '일', '시간']] = df_cat_tst['사고일시'].str.extract(time_pattern)
        df_cat_tst[['연', '월', '일', '시간']] = df_cat_tst[['연', '월', '일', '시간']].apply(pd.to_numeric)
        df_cat_tst = df_cat_tst.drop(columns=['사고일시'])

        location_pattern = r'(\S+) (\S+) (\S+)'
        
        df_cat[['도시', '구', '동']] = df_cat['시군구'].str.extract(location_pattern)
        df_cat = df_cat.drop(columns=['시군구'])

        df_cat_tst[['도시', '구', '동']] = df_cat_tst['시군구'].str.extract(location_pattern)
        df_cat_tst = df_cat_tst.drop(columns=['시군구'])

        road_pattern = r'(.+) - (.+)'
        
        df_cat[['도로형태1', '도로형태2']] = df_cat['도로형태'].str.extract(road_pattern)
        df_cat = df_cat.drop(columns=['도로형태'])

        df_cat_tst[['도로형태1', '도로형태2']] = df_cat_tst['도로형태'].str.extract(road_pattern)
        df_cat_tst = df_cat_tst.drop(columns=['도로형태'])
        
        ex_df = external_data()
        df_cat = pd.merge(df_cat, ex_df, how='left', on=['도시', '구', '동'])
        df_cat_tst = pd.merge(df_cat_tst, ex_df, how='left', on=['도시', '구', '동'])
        df_cat.to_csv('data.csv')
        encoder = Encoder_Module(df['ECLO'])
        df_cat = encoder.encoder(df_cat)
        df_cat_tst = encoder.encoder(df_cat_tst)

        df_cat.fillna(0, inplace=True)
        df_cat_tst.fillna(0, inplace=True)
        
        df = pd.concat([df_cat, df_num], axis=1)
        df_tst = pd.concat([df_cat_tst, df_tst_num], axis=1)
        
        return df, df_tst
    
    def __call__(self, df:pd.DataFrame):
        return