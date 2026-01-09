import requests
import re
from rdkit import Chem

from prompt.task_description import TASK_DESCRIPTION


def find_matches(text, k=3):
    # rag.jl endpoint
    url = "http://localhost:8003/find_matches"
    payload = {"query": text, "k": k}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from RAG service: {response.text}")

def normalize_author_name(name: str) -> str:
    """Normalize author name to 'lastname firstname_initials' format.

    Args:
        name: Raw author name string

    Returns:
        Normalized name (e.g., "hulterstrm a" for "A Hulterström")
    """
    step1 = re.sub(r"[^A-Za-z '\-]", "", name.strip())
    step2 = step1.lower()
    final_token = re.sub(r"\s+", " ", step2)
    last_first = final_token.split()[-1] + " " + ''.join([a[0] for a in final_token.split()[:-1]])
    
    return last_first

def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def extract_content(result) -> str:
    """Extract text content from agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            return str(msg.content)
    return ""

def get_task_description(task_name: str, seed_mol_index: int = 1, sim_threshold: float = 0.4) -> str:
    if 'lead_optimization' in task_name:
        protein_seedmol_dict = {'parp1': ['CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3', 'COc1[nH]c3cccc2C(=O)NCCc1c23', 'O/N=C/c1cn3CCNC(=O)c2cccc1c23'],
                                'fa7': ['CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34', 'N[C@H](Cc1ccccc1)C(=O)N2CCC[C@H]2C(=O)N[C@H](CCl)CCCN=C(N)N', 'CC(C)Nc3ccc(c1cc(N)cc(C(O)=O)c1)n(CC(=O)NCc2ccc(C(N)=N)cc2)c3=O'],
                                '5ht1b': ['Cc1nc(-c2ccc(-c3ccc(C(=O)N4CCc5cc6c(cc54)[C@]4(CC[N@H+](C)CC4)CO6)cc3)c(C)c2)no1', 'FC(F)(F)c1cccc(N2CC[NH2+]CC2)c1', 'C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1'],
                                'braf': ['CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34', 'FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)NCCN1CCOCC1)c2)cc3)ccc4Cl', 'FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)Nc1cccnc1)c2)cc3)ccc4Cl'],
                                'jak2': ['OCCCCc2nc1ccccc1c4ncnc3[nH]cc2c34', 'COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34', 'Oc5ccc(C2NC(=O)c1ccccc1c3ccnc4[nH]cc2c34)c(F)c5']}
        protein = task_name.split('/')[-1]
        seed_mol = protein_seedmol_dict[protein][seed_mol_index]
        
        task_description = TASK_DESCRIPTION[task_name].format(seed_mol=seed_mol, sim_threshold=sim_threshold)
    else:
        task_description = TASK_DESCRIPTION[task_name]
        
    return task_description