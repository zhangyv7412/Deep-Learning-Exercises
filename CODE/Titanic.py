from torch.utils.data import Dataset
import pandas as pd
import torch


class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274
        }

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) ##删除不用的列
        df = df.dropna(subset=["Age"])##删除Age有缺失的行
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)##进行one-hot编码

        ##进行数据的标准化
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)