import sys
import csv
import logging
from collections import defaultdict
from itertools import combinations
from collections import defaultdict, OrderedDict
import hashlib
from typing import List, Dict, DefaultDict, Optional, Any
import numpy as np
import numba
from collections import Counter
from bitarray import bitarray  # Core dependency, install with: pip install bitarray
import struct  # 用于精准控制字节序

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
        # ijtids_arr = np.array(self.tids, dtype=bool)
        # 步骤1：读取bitarray的底层字节缓冲区，转成uint8数组（每个元素是1个字节）
        bytes_arr = np.frombuffer(self.tids.tobytes(), dtype=np.uint8)
        # 步骤2：把每个字节解包成8个位（bit），得到完整的位数组
        bits_arr = np.unpackbits(bytes_arr)
        # 步骤3：截断到原bitarray的长度（去掉填充的0）
        ijtids_arr = bits_arr[:len(self.tids)].astype(bool)
        # ijtids_arr = np.array(self.tids.tolist(), dtype=bool)
        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
        # return sum((i + 1) for i, bit in enumerate(self.tids) if bit)
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

    def run(self):
        """Execute main CFD mining process"""
        singletons = self.get_singletons(self.min_supp)
        self.mine([], singletons, [])
        logger.info("len(self.generators): %s", len(self.generators))
        self.print_rules()

    def get_singletons(self, min_supp: int) -> list[MinerNode]:
        """Get single-item itemsets meeting minimum support (bitarray throughout)"""
        # Initialize: each item corresponds to a all-0 bitarray
        item_tids: Dict[int, TidList] = {}
        for item in self.db.frequencies.keys():
            item_tids[item] = bitarray(self.global_max_tid + 1)
            item_tids[item].setall(0)
        
        # Traverse transactions, set corresponding TID bits to 1
        for tid, row in enumerate(self.db.data):
            for item in row:
                item_tids[item][tid] = 1
        # Filter by minimum support, generate MinerNode
        singletons = []
        for item, tids in item_tids.items():
            supp = tids.count()
            if supp >= min_supp:
                # self.item_bitset_cache[item] = tids  # Cache bitarray
                singletons.append(MinerNode(item, tids, supp=supp))
        # Sort by support
        return sorted(singletons, key=lambda x: x.supp)

    # @profile
    def mine(self, prefix: Itemset, items: list[MinerNode], parent_closure: Itemset):
        """Recursively mine frequent itemsets and CFD rules (core logic)"""
        global index
        logger.debug("len(prefix): %s", len(prefix))
        # print("prefix:", prefix)
        if len(prefix) == self.max_size:
            return
        
        # Traverse items in reverse order
        # print("len(items):", len(items))
        for ix in reversed(range(len(items))):
            node = items[ix]
            iset = self.join_item(prefix, node.item)
            new_set = self.add_min_gen(GenMapEntry(iset, node.supp, node.hash))

            joins = []
            suffix = []

            # Branch: choose bucket method or direct intersection calculation
            if len(items) - ix - 1 > 2 * self.db.nr_attrs():
                # Bucket method (optimize intersection for large itemset counts)
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
                        # Fix: Compatible hash calculation
                        # hash_val = sum((i + 1) for i, bit in enumerate(ijtids) if bit)
                        # ijtids_arr = np.array(ijtids, dtype=bool)
                        ijtids_arr = np.array(ijtids.tolist(), dtype=bool)
                        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
                        suffix.append(MinerNode(jtem, ijtids, supp=ijsupp, hash_val=hash_val))
            else:
                # Direct intersection calculation (bitarray bitwise AND, optimal performance)
                for jx in range(ix + 1, len(items)):
                    j_node = items[jx]
                    global counter
                    counter += 1
                    # print("counter:", counter)
                    # Core: bitarray bitwise AND = intersection (native CPU operation)
                    ijtids = node.tids & j_node.tids
                    ijsupp = ijtids.count()
                    # print("ijsupp:", ijsupp)
                    # import pdb;pdb.set_trace()
                    if ijsupp == node.supp:
                        joins.append(j_node.item)
                    elif ijsupp >= self.min_supp:
                        # Fix: Compatible hash calculation
                        # hash_val = sum((i + 1) for i, bit in enumerate(ijtids) if bit)
                        # ijtids_arr = np.array(ijtids, dtype=bool)
                        # ijtids_arr = np.array(ijtids.tolist(), dtype=bool)
                        bytes_arr = np.frombuffer(ijtids.tobytes(), dtype=np.uint8)
                        bits_arr = np.unpackbits(bytes_arr)
                        ijtids_arr = bits_arr[:len(ijtids)].astype(bool)
                        hash_val = (np.where(ijtids_arr)[0] + 1).sum()
                        suffix.append(MinerNode(j_node.item, ijtids, supp=ijsupp, hash_val=hash_val))

            if joins:
                joins.sort()

            # Merge closures
            new_closure = self.join_sets(joins, parent_closure)
            cset = self.join_sets(iset, new_closure)
            post_min_gens = self.get_min_gens(cset, node.supp, node.hash)

            # Process minimal generators
            for ge in post_min_gens:
                if ge != new_set:
                    add = [x for x in cset if x not in ge.items]
                    if add:
                        uni = self.join_sets(add, ge.closure)
                        ge.closure = uni

            # Save generator
            if new_set:
                new_set.closure = new_closure
                items_hash = self.hash_itemset(new_set.items)
                if items_hash in self.generators:
                    import pdb;pdb.set_trace()
                self.generators[items_hash] = new_set
                index += 1
                logger.debug("index: %s", index)

            # Recursive mining
            if suffix:
                suffix.sort(key=lambda x: x.supp)
                self.mine(iset, suffix, new_closure)

    def print_rules(self):
        """Print mined CFD rules"""
        logger.info("len(self.generators): %s", len(self.generators))
        total_number = 0
        for gen in self.generators.values():
            items = gen.items
            rhs = gen.closure
            
            if not rhs:
                continue
            
            # Process antecedent conditions
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
            
            # Format output
            head_attrs = []
            head_vals = []
            for item in items:
                token = self.db.get_token(item)
                head_attrs.append(token.attr)
                head_vals.append(token.val)
            
            head_h = f"[{', '.join(head_attrs)}] => "
            head_v = f"({' ,'.join(head_vals)} || "
            
            for item in rhs:
                token = self.db.get_token(item)
                logger.info("%s%s, %s%s)", head_h, token.attr, head_v, token.val)
                total_number += 1
        logger.info("total_number: %s", total_number)
        logger.info("global counter: %d", counter)

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
    # @profile
    def add_min_gen(self, new_set: GenMapEntry) -> Optional[GenMapEntry]:
        """Add minimal generator (avoid redundancy)"""
        if new_set.hash in self.gen_map:
            # print("new_set.hash:", new_set.hash)
            # print("self.gen_map[new_set.hash]:", self.gen_map[new_set.hash])
            for g in self.gen_map[new_set.hash]:
                # print("new_set.supp:", new_set.supp)
                # print("g.supp:", g.supp)
                if g.supp == new_set.supp:
                    is_subset = self.is_subset(g.items, new_set.items)
                    is_equal = (g.items == new_set.items)
                    # print("is_subset:", is_subset)
                    # print("is_equal:", is_equal)
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
        # Initialize: each item corresponds to all-0 bitarray
        for jx in range(start_idx, len(items)):
            item = items[jx].item
            ijtid_map[item] = bitarray(self.global_max_tid + 1)
            ijtid_map[item].setall(0)
        
        # Fix: Use enumerate instead of itersearch to traverse 1 bits in node_tids
        for t, bit in enumerate(node_tids):
            if not bit:
                continue  # Only process bits with value 1 (corresponding TIDs)
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
        # Process headers
        headers = next(reader) if has_headers else [f"Attr{i+1}" for i in range(len(next(reader)))]
        
        # Read data rows
        if has_headers:
            data_rows = list(reader)
        else:
            f.seek(0)
            data_rows = list(reader)
        
        # Parse each row of data
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
    miner.run()

if __name__ == "__main__":
    main()