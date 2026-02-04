import requests
import re
from rdkit import Chem
import ast
import json
import logging
from typing import Any, List
import pandas as pd


from prompt.task_description import TASK_DESCRIPTION
from prompt.task_related_molecules import TASK_RELATED_MOLECULES

logger = logging.getLogger(__name__)

# Seed molecules for lead optimization tasks by protein target
PROTEIN_SEEDMOL_DICT = {
    'parp1': ['CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3', 'COc1[nH]c3cccc2C(=O)NCCc1c23', 'O/N=C/c1cn3CCNC(=O)c2cccc1c23'],
    'fa7': ['CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34', 'N[C@H](Cc1ccccc1)C(=O)N2CCC[C@H]2C(=O)N[C@H](CCl)CCCN=C(N)N', 'CC(C)Nc3ccc(c1cc(N)cc(C(O)=O)c1)n(CC(=O)NCc2ccc(C(N)=N)cc2)c3=O'],
    '5ht1b': ['Cc1nc(-c2ccc(-c3ccc(C(=O)N4CCc5cc6c(cc54)[C@]4(CC[N@H+](C)CC4)CO6)cc3)c(C)c2)no1', 'FC(F)(F)c1cccc(N2CC[NH2+]CC2)c1', 'C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1'],
    'braf': ['CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34', 'FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)NCCN1CCOCC1)c2)cc3)ccc4Cl', 'FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)Nc1cccnc1)c2)cc3)ccc4Cl'],
    'jak2': ['OCCCCc2nc1ccccc1c4ncnc3[nH]cc2c34', 'COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34', 'Oc5ccc(C2NC(=O)c1ccccc1c3ccnc4[nH]cc2c34)c(F)c5'],
    'sars_cov_2': ['C1=C(N=C(C(=O)N1)C(=O)N)F', 'CN(CC1=C(C(=CC(=C1)Br)Br)N)C2CCCCC2', 'CCC(C)SSC1=NC=CN1',
                   'C1=CC=C(C=C1)N2C(=O)C3=CC=CC=C3[Se]2', 'CCN(CC)C(=S)SSC(=S)N(CC)CC', 'C=C1[C@H](C[C@@H]([C@H]1CO)O)N2C=NC3=C2N=C(NC3=O)N',
                   'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O', 'C1=CC(=CC=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O']
}


def truncate_for_prompt(content: Any, max_chars: int = 50000) -> str:
    """Truncate content to fit within token limits.

    Args:
        content: The content to truncate (will be converted to string)
        max_chars: Maximum characters (roughly 4 chars per token)

    Returns:
        Truncated string representation
    """
    text = str(content)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} chars omitted]"


def find_matches(text, k=3):
    # rag.jl endpoint
    url = "http://localhost:8003/find_matches"
    k = min(k, 100)
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

def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def extract_content(result) -> str:
    """Extract text content from agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            return str(msg.content)
    return ""

def get_lead_optimization_mood_seed_molecules(protein: str) -> list:
    zinc_train_df = pd.read_csv('data/zinc250k_train.csv')
    seed_molecules = zinc_train_df.sort_values(by=f'{protein}/score', ascending=False).head(10)['smiles'].tolist()

    return seed_molecules
    
def get_task_description(task_name: str, seed_mol_index: int, sim_threshold: float) -> str:

    if 'lead_optimization' in task_name:
        protein = task_name.split('/')[-1]
        protein_name = {'parp1': 'PARP1', 'fa7': 'FA7', '5ht1b': '5-HT1B', 'braf': 'BRAF', 'jak2': 'JAK2',
                        'sars_cov_2': 'SARS-CoV-2'}.get(protein)
        # MOOD lead optimization
        if 'mood' in task_name:
            seed_molecules = get_lead_optimization_mood_seed_molecules(protein)
            docking_score_threshold = {'parp1': 10.0, 'fa7': 8.5, '5ht1b': 8.7845, 'braf': 10.3, 'jak2': 9.1, 'sars_cov_2': 10.0}.get(protein)
            task_description = TASK_DESCRIPTION[task_name].format(seed_molecules=seed_molecules, protein_name=protein_name, docking_score_threshold=docking_score_threshold)
        # Genmol lead optimization
        else:
            seed_mol = PROTEIN_SEEDMOL_DICT[protein][seed_mol_index]
            task_description = TASK_DESCRIPTION[task_name].format(seed_mol=seed_mol, sim_threshold=sim_threshold, protein_name=protein_name)

    elif 'boltz' in task_name:
        # Boltz binding affinity prediction task
        protein = task_name.split('/')[-1].upper()
        protein_sequence_dict = {'CA2': 'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK',
                            'TYK2': 'TVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADCGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDQGEKSLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRDLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPCEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYQ',
                            'CDK2': 'MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL',
                            'JNK1': 'MSRSKRDNNFYSVEIGDSTFTVLKRYQNLKPIGSGAQGIVCAAYDAILERNVAIKKLSRPFQNQTHAKRAYRELVLMKCVNHKNIIGLLNVFTPQKSLEEFQDVYIVMELMDANLCQVIQMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTAGTSFMMTPYVVTRYYRAPEVILGMGYKENVDIWSVGCIMGEMIKGGVLFPGTDHIDQWNKVIEQLGTPCPEFMKKLQPTVRTYVENRPKYAGYSFEKLFPDVLFPADSEHNKLKASQARDLLSKMLVIDASKRISVDEALQHPYINVWYDPSEAEAPPPKIPDKQLDEREHTIEEWKELIYKEVMDLEERTKNGVIRGQPSPLAQVQQ',
                            'P38': 'MSLIRKKGFYKQDVNKTAWELPKTYVSPTHVGSGAYGSVCSAIDKRSGEKVAIKKLSRPFQSEIFAKRAYRELLLLKHMQHENVIGLLDVFTPASSLRNFYDFYLVMPFMQTDLQKIMGMEFSEEKIQYLVYQMLKGLKYIHSAGVVHRDLKPGNLAVNEDCELKILDFGLARHADAEMTGYVVTRWYRAPEVILSWMHYNQTVDIWSVGCIMAEMLTGKTLFKGKDYLDQLTQILKVTGVPGTEFVQKLNDKAAKSYIQSLPQTPRKDFTQLFPRASPQAADLLEKMLELDVDKRLTAAQALTHPFFEPFRDPEEETEAQQPFDDSLEHEKLTVDEWKQHIYKEIVNFSPIARKDSRRRSGMKL',
                            'THROMBIN': 'IVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETWTANVGKGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE',
                            'DHFR': 'MVGSLNCIVAVSQNMGIGKNGDLPWPPLRNEFRYFQRMTTTSSVEGKQNLVIMGKKTWFSIPEKNRPLKGRINLVLSRELKEPPQGAHFLSRSLDDALKLTEQPELANKVDMVWIVGGSSVYKEAMNHPGHLKLFVTRIMQDFESDTFFPEIDLEKYKLLPEYPGVLSDVQEEKGIKYKFEVYEKND',
                            'FABP4': 'MCDAFVGTWKLVSSENFDDYMKEVGVGFATRKVAGMAKPNMIISVNGDVITIKSESTFKNTEISFILGQEFDEVTADDRKVKSTITLDGGVLVHVQKWDGKSTTIKRKREDDKLVVECVMKGVTSTRVYERA'}
        task_description = TASK_DESCRIPTION[task_name].format(protein_name=protein, protein_sequence=protein_sequence_dict.get(protein))

    else:
        task_description = TASK_DESCRIPTION[task_name]

    return task_description

def safe_parse_json_list(text: str) -> list:
    """Safely parse JSON/Python list from LLM response.

    Tries multiple strategies to handle malformed/truncated responses:
    1. Direct ast.literal_eval
    2. Extract JSON array with regex and parse with json.loads
    3. Fix truncated responses by finding last complete entry
    4. Extract individual complete objects with regex

    Args:
        text: The LLM response text to parse

    Returns:
        Parsed list of dictionaries, or empty list if parsing fails
    """
    if not text or not text.strip():
        return []

    # Strategy 1: Try direct parsing
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass

    # Strategy 2: Extract JSON array with regex
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Fix truncated responses - find last complete entry
    try:
        fixed = text
        # Remove trailing commas before ] or }
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

        # Find the last complete JSON object by looking for "},"
        last_complete = fixed.rfind('},')
        if last_complete != -1:
            fixed = fixed[:last_complete + 1] + ']'

        # Close unclosed brackets
        open_brackets = fixed.count('[') - fixed.count(']')
        fixed += ']' * max(0, open_brackets)
        open_braces = fixed.count('{') - fixed.count('}')
        fixed += '}' * max(0, open_braces)

        return ast.literal_eval(fixed)
    except (SyntaxError, ValueError):
        pass

    # Strategy 4: Last resort - find all complete SMILES objects individually
    try:
        pattern = r'\{\s*"SMILES"\s*:\s*"[^"]+"\s*(?:,\s*"[^"]+"\s*:\s*(?:"[^"]*"|[^,}]+)\s*)*\}'
        matches = re.findall(pattern, text)
        if matches:
            results = []
            for m in matches:
                try:
                    results.append(ast.literal_eval(m))
                except (SyntaxError, ValueError):
                    continue
            if results:
                return results
    except Exception:
        pass

    # If all parsing fails, return empty list and log warning
    logger.warning(f"Failed to parse LLM response: {text[:200]}...")
    return []

def get_related_molecules(task_name: str, seed_mol_index: int = 1) -> str:
    if 'lead_optimization' in task_name:
        protein = task_name.split('/')[-1]
        related_molecules = [PROTEIN_SEEDMOL_DICT[protein][seed_mol_index]]
    else:
        related_molecules = TASK_RELATED_MOLECULES[task_name]
        
    return related_molecules

def extract_smiles(text: str) -> List[str]:
    """Extract SMILES strings from text.

    Uses safe parsing to handle malformed/truncated LLM responses.
    """
    result = safe_parse_json_list(text)
    smiles_list = []
    for r in result:
        if isinstance(r, dict) and 'SMILES' in r:
            canonical = canonicalize_smiles(r['SMILES'])
            if canonical:
                smiles_list.append(canonical)
    return smiles_list