from .DataPipelineBase import DataPipelineBase
from typing import Any


class DataPipelineClicks(DataPipelineBase):
    # def __init__(self, dataframe_path: str) -> None:
    #     super().__init__(dataframe_path)

    # def __call__(self) -> Any:
    #     df = self..merge_data(
    #         df_to_merge=pubs_df_path, 
    #         left_on='publication_id',
    #         right_on='id',
    #         output_features=['user_id', 'publication_id']
    #     )

    #     ds = pipeline.convert_to_tf_dataset(df)

    #     vocabularies = pipeline.build_vocabularies(
    #         features=['user_id', 'publication_id'], ds=ds, batch=1_000)
        
    #     total, train_Length, test_length = pipeline.get_lengths(ds)

    #     print(pipeline)