from functools import partial
from typing import List, Dict, Any, Union, Optional, Tuple

import datasets

datasets.disable_caching()


def get_torch_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset.set_format(type="torch")
    return dataset


def _merge_scene_uncanny_caption(
    data_instances: Dict[str, List[Any]],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str,
) -> Dict[str, List[Any]]:
    """See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.data_processing.utils.html."""
    # TODO-2.1

    dataset = datasets.Dataset.from_dict(data_instances)
    def merge_train_val(x):
      templated_text = scene_colname_and_special_token[1] + " " + \
      x[scene_colname_and_special_token[0]] + " " + \
      uncanny_colname_and_special_token[1] + " " + \
      x[uncanny_colname_and_special_token[0]] + " " + \
      caption_colname_and_special_token[1] + " " + \
      x[caption_colname_and_special_token[0]] + " " + \
      end_of_caption_special_token

      # print(templated_text)
      x[merge_colname] = templated_text
      return x

    def merge_test(x):
      templated_text = scene_colname_and_special_token[1] + " " + \
      x[scene_colname_and_special_token[0]] + " " + \
      uncanny_colname_and_special_token[1] + " " + \
      x[uncanny_colname_and_special_token[0]] + " " + \
      caption_colname_and_special_token[1]
      
      # print(templated_text)
      x[merge_colname] = templated_text
      return x


    if caption_colname_and_special_token[0] in dataset.column_names:
      # train, val
      dataset = dataset.map(merge_train_val)
    else:
      # test
      dataset = dataset.map(merge_test)
    return dataset.to_dict()

    
def generate_newyorker_lm_text_dataset(
    newyorker_dataset: Union[datasets.Dataset, datasets.dataset_dict.DatasetDict],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str = "text",
    batch_size: int = 4000,
    remove_cols: Optional[list] = None,
) -> Union[datasets.Dataset, datasets.dataset_dict.DatasetDict]:
    formatting_fn = partial(
        _merge_scene_uncanny_caption,
        scene_colname_and_special_token=scene_colname_and_special_token,
        uncanny_colname_and_special_token=uncanny_colname_and_special_token,
        caption_colname_and_special_token=caption_colname_and_special_token,
        end_of_caption_special_token=end_of_caption_special_token,
        merge_colname=merge_colname,
    )
    newyorker_dataset = newyorker_dataset.map(formatting_fn, batched=True, batch_size=batch_size).shuffle(seed=4740)
    if remove_cols is not None:
        newyorker_dataset = newyorker_dataset.remove_columns(remove_cols)
    return newyorker_dataset
