import sys
import csv
import logging
from collections import defaultdict
from itertools import combinations
from collections import defaultdict, OrderedDict
import hashlib
from typing import List, Dict, DefaultDict, Optional, Any, Union, Set
import numpy as np
import numba
from collections import Counter
from bitarray import bitarray  # Core dependency, install with: pip install bitarray
import struct
import pandas as pd
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

# ===================== Log Configuration =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ===================== Type Definitions: TidList changed to bitarray (Core) =====================
Transaction = list[int]
Itemset = list[int]
TidList = bitarray  # Replace original list[int]

# Global variables
index = 0
counter = 0

# ===================== DbToken Class (No modifications) =====================
class DbToken:
    def __init__(self, attr: str, val: str):
        self.attr = attr
        self.val = val

    def __hash__(self) -> int:
        return hash((self.attr, self.val))

    def __eq__(self, other) -> bool:
        if not isinstance(other, DbToken):
            return False
        return self.attr == other.attr and self.val == other.val

    def __repr__(self) -> str:
        return f"{self.attr}={self.val}"

# ===================== Database Class (No modifications) =====================
class Database:
    def __init__(self):
        self.data: list[Transaction] = []  # Transaction data (item list)
        self.token_to_int: dict[DbToken, int] = {}
        self.int_to_token: dict[int, DbToken] = {}
        self.attr_map: dict[int, int] = {}
        self.frequencies: dict[int, int] = defaultdict(int)
        self.attributes: list[str] = []
        self.attr_counter = 1
        self.num_tokens = 1

    def size(self) -> int:
        return len(self.data)

    def nr_attrs(self) -> int:
        return len(self.attributes)

    def nr_items(self) -> int:
        return len(self.token_to_int)

    def get_row(self, row: int) -> Transaction:
        return self.data[row]

    def get_token(self, i: int) -> DbToken:
        return self.int_to_token[i]

    def add_row(self, row: Transaction):
        self.data.append(row)

    def translate_token(self, token: DbToken) -> int:
        if token in self.token_to_int:
            return self.token_to_int[token]
        
        if token.attr not in self.attributes:
            self.attributes.append(token.attr)
        
        attr_id = self.attributes.index(token.attr) + 1
        self.token_to_int[token] = self.num_tokens
        self.int_to_token[self.num_tokens] = token
        self.attr_map[self.num_tokens] = attr_id
        
        token_id = self.num_tokens
        self.num_tokens += 1
        return token_id

    def inc_freq(self, item: int):
        self.frequencies[item] += 1

    def frequency(self, item: int) -> int:
        return self.frequencies.get(item, 0)
    @classmethod
    def from_dataframe_subset(cls, df: pd.DataFrame, columns: List[str]) -> "Database":
        """

        """
        db = cls()
        #
        subset_df = df[columns].dropna()
        
        for _, row in subset_df.iterrows():
            transaction = []
            for col in columns:
                val = str(row[col]).strip()
                if val == "":
                    continue
                token = DbToken(attr=col, val=val)
                item_id = db.translate_token(token)
                db.inc_freq(item_id)
                transaction.append(item_id)
            if transaction:  #
                db.add_row(transaction)
        
        logger.info(f"complete：col={columns}，transactions={db.size()}，token={db.nr_items()}")
        return db

# ===================== MinerNode Class (Fix itersearch compatibility) =====================
class MinerNode:
    def __init__(self, item: int, tids: TidList = None, supp: int = 0, hash_val: int = 0):
        self.item = item
        # Initialize bitarray, empty bitarray by default (length set later)
        self.tids = tids if tids is not None else bitarray()
        # supp = number of 1 bits in bitarray (transaction count)
        self.supp = supp if supp != 0 else self.tids.count()
        # Calculate hash (compatible with all bitarray versions)
        self.hash = hash_val if hash_val != 0 else self._hash_tids()

    def _hash_tids(self) -> int:
        """Fix: Use enumerate instead of itersearch for full bitarray compatibility"""
        bytes_arr = np.frombuffer(self.tids.tobytes(), dtype=np.uint8)
        bits_arr = np.unpackbits(bytes_arr)
        ijtids_arr = bits_arr[:len(self.tids)].astype(bool)
        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
        return hash_val

    def __repr__(self) -> str:
        """Fix: Compatible format, convert to TID list for debugging"""
        tid_list = [i for i, bit in enumerate(self.tids) if bit]
        return f"MinerNode(item={self.item}, supp={self.supp}, tids={tid_list})"

# ===================== GenMapEntry Class (No modifications) =====================
class GenMapEntry:
    def __init__(self, items: Itemset, supp: int, hash_val: int = 0):
        self.items = items
        self.closure = []
        self.supp = supp
        self.hash = hash_val

# ===================== Rule & CandidateRule Data Classes (Updated) =====================
class RuleDetail(BaseModel):
    rule: str = Field(
        ...,
        description='A natural language expression of the candidate rule.'
    )
    explanation: str = Field(
        ...,
        description='A clear explanation of the logic and rationale behind the rule'
    )
    columns: Set[str] = Field(
        ...,
        description='A set of unique column names mentioned in this rule, '
                   'indicating which columns are referenced in this rule. '
                   'Note that this field should be specified regardless of possible specifications of columns '
                   'in the rule field.'
    )

@dataclass
class CandidateRule:
    rule_id: Union[int, str]
    rule_type: str
    rule: RuleDetail
    code: Optional[str] = field(default=None)
    execution_result: Optional[Dict[str, Union[int, float, List[int]]]] = field(default=None)
    semantic_validity: Optional[bool] = field(default=None)

    def __str__(self):
        return (
            f"CandidateRule(rule_id={self.rule_id}, "
            f"rule details: {self.rule.rule}, execution_result=omitted)"
        )

# ===================== CCFDMiner Class (Core refactoring + compatibility fixes) =====================
class CCFDMiner:
    def __init__(self, db: Database, min_supp: int, max_size: int):
        self.db = db
        self.min_supp = min_supp
        self.max_size = max_size
        self.gen_map: dict[int, list[GenMapEntry]] = defaultdict(list)
        self.generators: dict[int, GenMapEntry] = {}
        self.global_max_tid = self.db.size() - 1  # Global maximum TID (ensure consistent bitarray length)
        self.item_bitset_cache: dict[int, TidList] = {}  # Cache for item bitarrays

    def run(self, df: pd.DataFrame) -> List[CandidateRule]:
        """Execute main CFD mining process and return structured rules"""
        singletons = self.get_singletons(self.min_supp)
        self.mine([], singletons, [])
        logger.info("len(self.generators): %s", len(self.generators))
        return self.print_rules(df)

    def get_singletons(self, min_supp: int) -> list[MinerNode]:
        """Get single-item itemsets meeting minimum support (bitarray throughout)"""
        item_tids: Dict[int, TidList] = {}
        for item in self.db.frequencies.keys():
            item_tids[item] = bitarray(self.global_max_tid + 1)
            item_tids[item].setall(0)
        
        for tid, row in enumerate(self.db.data):
            for item in row:
                item_tids[item][tid] = 1
        
        singletons = []
        for item, tids in item_tids.items():
            supp = tids.count()
            if supp >= min_supp:
                singletons.append(MinerNode(item, tids, supp=supp))
        return sorted(singletons, key=lambda x: x.supp)

    def mine(self, prefix: Itemset, items: list[MinerNode], parent_closure: Itemset):
        """Recursively mine frequent itemsets and CFD rules (core logic)"""
        global index
        if len(prefix) == self.max_size:
            return
        
        for ix in reversed(range(len(items))):
            node = items[ix]
            iset = self.join_item(prefix, node.item)
            new_set = self.add_min_gen(GenMapEntry(iset, node.supp, node.hash))

            joins = []
            suffix = []

            if len(items) - ix - 1 > 2 * self.db.nr_attrs():
                ijtid_map = self.bucket_tids(items, ix + 1, node.tids)
                for jx in range(ix + 1, len(items)):
                    jtem = items[jx].item
                    if jtem not in ijtid_map:
                        continue
                    ijtids = ijtid_map[jtem]
                    ijsupp = ijtids.count()
                    
                    if ijsupp == node.supp:
                        joins.append(jtem)
                    elif ijsupp >= self.min_supp:
                        bytes_arr = np.frombuffer(ijtids.tobytes(), dtype=np.uint8)
                        bits_arr = np.unpackbits(bytes_arr)
                        ijtids_arr = bits_arr[:len(ijtids)].astype(bool)
                        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
                        suffix.append(MinerNode(jtem, ijtids, supp=ijsupp, hash_val=hash_val))
            else:
                for jx in range(ix + 1, len(items)):
                    j_node = items[jx]
                    global counter
                    counter += 1
                    ijtids = node.tids & j_node.tids
                    ijsupp = ijtids.count()
                    
                    if ijsupp == node.supp:
                        joins.append(j_node.item)
                    elif ijsupp >= self.min_supp:
                        bytes_arr = np.frombuffer(ijtids.tobytes(), dtype=np.uint8)
                        bits_arr = np.unpackbits(bytes_arr)
                        ijtids_arr = bits_arr[:len(ijtids)].astype(bool)
                        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
                        suffix.append(MinerNode(j_node.item, ijtids, supp=ijsupp, hash_val=hash_val))

            if joins:
                joins.sort()

            new_closure = self.join_sets(joins, parent_closure)
            cset = self.join_sets(iset, new_closure)
            post_min_gens = self.get_min_gens(cset, node.supp, node.hash)

            for ge in post_min_gens:
                if ge != new_set:
                    add = [x for x in cset if x not in ge.items]
                    if add:
                        uni = self.join_sets(add, ge.closure)
                        ge.closure = uni

            if new_set:
                new_set.closure = new_closure
                items_hash = self.hash_itemset(new_set.items)
                if items_hash in self.generators:
                    import pdb;pdb.set_trace()
                self.generators[items_hash] = new_set
                index += 1
                logger.debug("index: %s", index)

            if suffix:
                suffix.sort(key=lambda x: x.supp)
                self.mine(iset, suffix, new_closure)

    def print_rules(self, df: pd.DataFrame) -> List[CandidateRule]:
        """Print mined CFD rules and return them as a list of CandidateRule instances"""
        logger.info("len(self.generators): %s", len(self.generators))
        total_number = 0
        rules_list = []
        for gen in self.generators.values():
            items = gen.items
            rhs = gen.closure
            
            if not rhs:
                continue
            
            for leave_out in items:
                sub = self.subset(items, leave_out)
                sub_hash = self.hash_itemset(sub)
                if sub_hash in self.generators:
                    sub_closure = self.generators[sub_hash].closure
                    diff = [x for x in rhs if x not in sub_closure]
                    if not diff:
                        rhs = []
                        break
                    rhs = diff
            
            if not rhs:
                continue
            
            head_attrs = []
            head_vals = []
            for item in items:
                token = self.db.get_token(item)
                head_attrs.append(token.attr)
                head_vals.append(token.val)
            
            for item in rhs:
                token = self.db.get_token(item)
                rule_info = {
                    "antecedent_attrs": head_attrs,
                    "antecedent_vals": head_vals,
                    "consequent_attr": token.attr,
                    "consequent_val": token.val,
                    "support": gen.supp
                }
                rules_list.append(rule_info)
                logger.info("%s[%s] => %s (%s || %s)", 
                            ", ".join(head_attrs), 
                            ", ".join(head_vals), 
                            token.attr, 
                            ", ".join(head_vals), 
                            token.val)
                total_number += 1
        logger.info("total_number: %s", total_number)
        logger.info("global counter: %d", counter)
        

        return convert_cfd_rules_to_candidate_rules(rules_list, df)

    @staticmethod
    def join_item(itemset: Itemset, item: int) -> Itemset:
        """Add single item to itemset and sort"""
        new_set = itemset.copy()
        new_set.append(item)
        new_set.sort()
        return new_set

    @staticmethod
    def join_sets(set1: Itemset, set2: Itemset) -> Itemset:
        """Merge two sorted itemsets and remove duplicates"""
        merged = []
        i = j = 0
        while i < len(set1) and j < len(set2):
            if set1[i] < set2[j]:
                merged.append(set1[i])
                i += 1
            elif set1[i] > set2[j]:
                merged.append(set2[j])
                j += 1
            else:
                merged.append(set1[i])
                i += 1
                j += 1
        merged.extend(set1[i:])
        merged.extend(set2[j:])
        return merged

    @staticmethod
    def subset(itemset: Itemset, leave_out: int) -> Itemset:
        """Generate subset excluding specified item"""
        return [x for x in itemset if x != leave_out]

    @staticmethod
    def hash_itemset(itemset: Itemset) -> int:
        """Calculate itemset hash value (collision resistant)"""
        hash_obj = hashlib.md5()
        for item in itemset:
            item_bytes = item.to_bytes(4, byteorder='big', signed=True)
            hash_obj.update(item_bytes)
        hash_hex = hash_obj.hexdigest()
        hash_val = int(hash_hex, 16)
        return hash_val
    
    def add_min_gen(self, new_set: GenMapEntry) -> Optional[GenMapEntry]:
        """Add minimal generator (avoid redundancy)"""
        if new_set.hash in self.gen_map:
            for g in self.gen_map[new_set.hash]:
                if g.supp == new_set.supp:
                    is_subset = self.is_subset(g.items, new_set.items)
                    is_equal = (g.items == new_set.items)
                    if is_subset and not is_equal:
                        return None
                    elif is_equal:
                        return None

        self.gen_map[new_set.hash].append(new_set)
        return self.gen_map[new_set.hash][-1]

    def get_min_gens(self, items: Itemset, supp: int, hash_val: int) -> list[GenMapEntry]:
        """Get minimal generators meeting criteria"""
        min_gens = []
        if hash_val in self.gen_map:
            for g in self.gen_map[hash_val]:
                if g.supp == supp and self.is_subset(g.items, items):
                    min_gens.append(g)
        return min_gens

    def bucket_tids(self, items: list[MinerNode], start_idx: int, node_tids: TidList) -> dict[int, TidList]:
        """Bucket method to optimize intersection calculation (fix itersearch compatibility)"""
        ijtid_map: Dict[int, TidList] = {}
        for jx in range(start_idx, len(items)):
            item = items[jx].item
            ijtid_map[item] = bitarray(self.global_max_tid + 1)
            ijtid_map[item].setall(0)
        
        for t, bit in enumerate(node_tids):
            if not bit:
                continue
            tup = self.db.get_row(t)
            for item in tup:
                if item in ijtid_map:
                    ijtid_map[item][t] = 1
        
        return ijtid_map

    @staticmethod
    def is_subset(sub: Itemset, super_set: Itemset) -> bool:
        """Check if sub is a subset of super_set"""
        return all(x in super_set for x in sub)

# ===================== Data Loading Function =====================
def load_csv(file_path: str, delim: str = ',', has_headers: bool = True) -> Database:
    """Load CSV file into Database object"""
    db = Database()
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delim)
        headers = next(reader) if has_headers else [f"Attr{i+1}" for i in range(len(next(reader)))]
        
        if has_headers:
            data_rows = list(reader)
        else:
            f.seek(0)
            data_rows = list(reader)
        
        for row in data_rows:
            if not row:
                continue
            transaction = []
            for i, val in enumerate(row):
                if i >= len(headers):
                    continue
                token = DbToken(headers[i], val.strip())
                item = db.translate_token(token)
                db.inc_freq(item)
                transaction.append(item)
            db.add_row(transaction)
    
    return db

# ===================== Code Generation & Conversion Functions =====================
def generate_rule_validate_code(
    antecedent_attrs: List[str],
    antecedent_vals: List[str],
    consequent_attr: str,
    consequent_val: str,
    comparison_operator: str = '=='
) -> str:
    if len(antecedent_attrs) == 1:
        condition_str = f"df['{antecedent_attrs[0]}'] == {antecedent_vals[0]}"
        rule_desc = f"If {antecedent_attrs[0]} = {antecedent_vals[0]}, then {consequent_attr} {comparison_operator} {consequent_val}."
    else:
        condition_parts = [f"(df['{attr}'] == {val})" for attr, val in zip(antecedent_attrs, antecedent_vals)]
        condition_str = " & ".join(condition_parts)
        antecedent_str = ", ".join([f"{attr} = {val}" for attr, val in zip(antecedent_attrs, antecedent_vals)])
        rule_desc = f"If {antecedent_str}, then {consequent_attr} {comparison_operator} {consequent_val}."

    code_template = '''import pandas as pd

def validate_rule(df):
    """
    Rule: {rule_desc}
    
    - <condition_column>: The column to check for the condition.
    - <condition_value>: The specific value in the condition column that triggers the rule.
    - <target_column>: The column whose values are being validated based on the rule.
    - <comparison_operator>: The logical operator (e.g., '==', '!=', 'is not None', etc.).
    - <target_value>: The value to compare the target column against.
    """

    # Filter rows where the condition is met
    condition_rows = df[{condition_str}]

    # Identify violations based on the target column and comparison
    violations = condition_rows[~(condition_rows['{consequent_attr}'] {comparison_operator} '{consequent_val}')].index.tolist()

    # Identify rows that satisfy the rule
    satisfactions = condition_rows[(condition_rows['{consequent_attr}'] {comparison_operator} '{consequent_val}')].index.tolist()

    # Calculate support and confidence
    total_condition = len(condition_rows)
    support = len(satisfactions) / len(df) if len(df) > 0 else 0
    confidence = len(satisfactions) / total_condition if total_condition > 0 else 0

    return {{
        "support": support,
        "confidence": confidence,
        "satisfactions": satisfactions,
        "violations": violations,
        "total_satisfactions": len(satisfactions),
        "total_violations": len(violations),
    }}'''
    
    code = code_template.format(
        rule_desc=rule_desc,
        condition_str=condition_str,
        consequent_attr=consequent_attr,
        comparison_operator=comparison_operator,
        consequent_val=consequent_val
    )
    return code

def execute_validate_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """

    
    Args:
        code
        df
    
    Returns:

    """
    local_namespace = {}
    try:
        exec(code, globals(), local_namespace)
        validate_func = local_namespace['validate_rule']
        result = validate_func(df)
        return result
    except Exception as e:
        logger.error(f"failure：{e}")
        return {
            "support": 0.0,
            "confidence": 0.0,
            "satisfactions": [],
            "violations": [],
            "total_satisfactions": 0,
            "total_violations": 0,
        }

def convert_cfd_rules_to_candidate_rules(cfd_rules: List[Dict[str, Any]], df: pd.DataFrame) -> List[CandidateRule]:
    """
    Convert CFD rules from CCFDMiner to CandidateRule instances.
    
    Args:
        cfd_rules: List of rule dictionaries from CCFDMiner.print_rules()
        
    Returns:
        List of CandidateRule instances
    """
    candidate_rules = []
    for idx, rule_dict in enumerate(cfd_rules):
        rule_id = idx + 1
        rule_type = "cfd_rule"
        
        antecedent_attrs = rule_dict["antecedent_attrs"]
        antecedent_vals = rule_dict["antecedent_vals"]
        consequent_attr = rule_dict["consequent_attr"]
        consequent_val = rule_dict["consequent_val"]
        support_count = rule_dict["support"]
        
        if len(antecedent_attrs) == 1:
            rule_str = f"If {antecedent_attrs[0]} = '{antecedent_vals[0]}', then {consequent_attr} = '{consequent_val}'."
        else:
            antecedent_str = ", ".join([f"{attr} = '{val}'" for attr, val in zip(antecedent_attrs, antecedent_vals)])
            rule_str = f"If {antecedent_str}, then {consequent_attr} = '{consequent_val}'."
        
        explanation_str = ""
        
        columns_set = set(antecedent_attrs + [consequent_attr])
        
        rule_detail = RuleDetail(
            rule=rule_str,
            explanation=explanation_str,
            columns=columns_set
        )
        
        validate_code = generate_rule_validate_code(
            antecedent_attrs=antecedent_attrs,
            antecedent_vals=antecedent_vals,
            consequent_attr=consequent_attr,
            consequent_val=consequent_val,
            comparison_operator='=='
        )
        
        validate_result = execute_validate_code(validate_code, df)
        
        # 构建execution_result（使用真实的行ID列表）
        execution_result = {
            "rule_id": rule_id,
            "support": validate_result["support"],
            "confidence": validate_result["confidence"],
            "satisfactions": validate_result["satisfactions"],  # 真实的行ID列表
            "violations": validate_result["violations"],        # 真实的行ID列表
            "total_satisfactions": validate_result["total_satisfactions"],
            "total_violations": validate_result["total_violations"],
            "mined_support_count": support_count  # 保留挖掘得到的支持度计数
        }
        
        candidate_rule = CandidateRule(
            rule_id=rule_id,
            rule_type=rule_type,
            rule=rule_detail,
            code=validate_code,
            execution_result=execution_result,
            semantic_validity=None
        )
        
        candidate_rules.append(candidate_rule)
    
    return candidate_rules

# ===================== Main Function =====================
def main():
    """Command line entry point"""
    if len(sys.argv) != 4:
        logger.info("Usage: python cfd_bitset.py infile minsupp maxsize")
        logger.info("\twhere infile is a CSV-format dataset, minsupp is a positive integer representing minimum support,")
        logger.info("\tmaxsize is a positive integer representing the maximum size of rules (maximum number of attributes in a rule)")
        return

    infile = sys.argv[1]
    try:
        min_supp = int(sys.argv[2])
        max_size = int(sys.argv[3])
    except ValueError:
        logger.error("minsupp and maxsize must be integers")
        return

    # Load data
    logger.info("Loading data file: %s", infile)
    db = load_csv(infile)
    logger.info("Data loading completed, total %s rows, %s attributes", db.size(), db.nr_attrs())

    # Run CFD mining
    logger.info("Start mining, minimum support: %s, maximum rule size: %s", min_supp, max_size)
    miner = CCFDMiner(db, min_supp, max_size)
    df = pd.read_csv(infile)
    candidate_rules = miner.run(df)
    
    for cr in candidate_rules:
        print(f"\n=== Rule ID: {cr.rule_id} ===")
        print(f"Rule (natural language): {cr.rule.rule}")
        print(f"Explanation: {cr.rule.explanation}")
        print(f"Columns involved: {cr.rule.columns}")
        print(f"Execution Result: {cr.execution_result}")
        print(f"code: {cr.code}")

if __name__ == "__main__":
    main()