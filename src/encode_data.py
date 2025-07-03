from typing import Optional, Dict, Any

import mteb
from huggingface_hub import HfApi

def get_last_commit_id(llm_model: str) -> Optional[str]:
    """ Gets last commit id from llm model specified

    Parameters
    ----------
      llm_model: str
        Model name

    Returns
    -------
      commit_id: str
        Commit id

    """
    commit_id = None

    api = HfApi()
    refs = api.list_repo_refs(llm_model)
    for branch in refs.branches:
        if branch.name == 'main':
            commit_id = branch.target_commit

    return commit_id


def encode(text: Dict[str, str], key_to_encode: str, 
           model_name: str, commit_id: str,
    batch_size: int = 32) -> Dict[str, Any]:
    """
    Encode text column using llm model

    Parameters
    ----------
    text: Dict[str, str]
      Dictionary with text to encode
    key_to_encode: str
        Key to encode in the dictionary
    model_name: str
      Model to use for encoding
    commit_id: str
      Commit id of the model to use
    batch_size: int
      Batch size to use for encoding

    Returns
    -------
    encoded_text: dictionary
      DataFrame with encoded text as numerical features

    """

    try:
        # load model using registry implementation if available
        model = mteb.get_model(model_name, commit_id)

        encoded_text = model.encode(text, batch_size=batch_size)
        
        return encoded_text

    except ValueError as e:
        return None


