import jax

from ..utils import jax_utils, Pytree


class ProcessRecords:
    """
    A class holding a sequence of records of one process state.

    Records are split into several chunks that are joined on-demand into one final chunk.
    Every chunk is a Pytree (resp. PropTree), where every array has a new leading dimension for
    the sequence in history.

    Optionally, only chunks divisible by `stride` are kept in history (they are still computed
    in-process and passed to this object, though).
    """

    def __init__(self, stride: int = 1) -> None:
        self.chunks = []
        self.stride = stride
        # Number of stride-1 records (=steps) we have seen
        self.steps = 0
        # Number of records kept, also the len
        self.records = 0

    def __len__(self):
        return self.records

    def copy(self):
        """
        Creates a copy, sharing all the pytrees with the original (assumes none are modified later).
        """
        s = ProcessRecords(self.stride)
        s.chunks = list(self.chunks)
        s.steps = self.steps
        s.records = self.records
        return s

    def last_record(self):
        """
        Return pytree of the last record (without the leading record axis)
        """
        if len(self.chunks) == 0:
            raise ValueError(f"No records in {self!r}")
        return jax.tree_util.tree_map(lambda a: a[-1], self.chunks[-1])

    def all_records(self):
        """
        Return the concatenated records as a pytree.

        Fails if no records were recorded so far.
        """
        if len(self.chunks) == 0:
            raise ValueError(f"No records in {self!r}")
        merged = jax_utils.concatenate_pytrees(self.chunks)
        assert self._chunk_len(merged) == self.records
        return merged

    @classmethod
    def _chunk_len(_cls, chunk: Pytree) -> int:
        ls = jax.tree_util.tree_leaves(chunk)
        if not ls:
            raise ValueError(f"No leaves in Pytree, can't measure chunk length")
        return ls[0].shape[0]

    def add_record(self, records: Pytree):
        # Skip records without leaves
        if len(jax.tree_util.tree_leaves(records)) == 0:
            return
        # number of new records
        l = self._chunk_len(records)
        if l == 0:
            return
        # first new record to take
        m = (self.stride - self.steps) % self.stride
        if m < l:
            strided = jax.tree_util.tree_map(lambda a: a[m :: self.stride], records)
            print(strided)
            self.records += self._chunk_len(strided)
            self.chunks.append(strided)
        self.steps += l

    def block_on_all(self):
        """
        Block until all arrays are actually computed (does not copy them to CPU).
        """
        if len(self.chunks) > 0:
            for v in jax.tree_util.tree_leaves(self.chunks[-1]):
                v.block_until_ready()
