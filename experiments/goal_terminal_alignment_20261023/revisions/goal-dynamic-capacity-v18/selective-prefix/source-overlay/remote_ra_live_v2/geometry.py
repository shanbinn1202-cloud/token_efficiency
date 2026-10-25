"""Allocation IDs are canonical; backend kernel IDs never own BlockPool refs.

Expansion follows captured BlockTable and the loaded B/K geometry, never a
hard-coded kernel width. TensorIO independently validates actual storage stride.
"""
from dataclasses import dataclass
from .contract import require


@dataclass(frozen=True)
class AllocationId:
    group: int
    value: int

    def __post_init__(self):
        require(type(self.group) is int and self.group >= 0 and
                type(self.value) is int and self.value > 0,
                'positive allocation-domain ID required; null is not a lease')

    @classmethod
    def from_wire(cls, group, value):
        # No int() coercion: bool/float/kernel-domain objects are not pool IDs.
        return cls(group, value)


@dataclass(frozen=True)
class KernelId:
    group: int
    value: int

    def __post_init__(self):
        require(type(self.group) is int and self.group >= 0 and
                type(self.value) is int and self.value >= 0,
                'kernel-domain ID required')


@dataclass(frozen=True)
class GroupGeometry:
    group: int
    kind: str
    allocation_tokens: int
    storage_tokens: int
    kernel_tokens: int
    allocation_capacity: int

    def __post_init__(self):
        require(type(self.group) is int and self.group >= 0 and
                self.kind in ('attention', 'hybrid'), 'invalid geometry identity')
        require(all(type(v) is int and v > 0 for v in
                    (self.allocation_tokens, self.storage_tokens,
                     self.kernel_tokens, self.allocation_capacity)),
                'positive exact geometry required')

    def allocation(self, value):
        ref = AllocationId.from_wire(self.group, value)
        require(ref.value < self.allocation_capacity, 'allocation exceeds pool capacity')
        return ref

    @property
    def ratio(self):
        require(self.storage_tokens == self.allocation_tokens and
                self.allocation_tokens % self.kernel_tokens == 0,
                'unsupported compressed/nonintegral allocation geometry')
        return self.allocation_tokens // self.kernel_tokens

    def expand(self, ref, validator=None):
        require(type(ref) is AllocationId and ref.group == self.group,
                'expansion requires same-group allocation ID')
        self.allocation(ref.value)
        require(self.kind == 'attention', 'Mamba states do not expand to attention kernels')
        require(callable(validator), 'installed backend expansion validator required')
        # Validator must be bound to the exact loaded backend/source evidence.
        # Generic arithmetic here must not be mistaken for loaded validation.
        ids = validator(self, ref)
        require(type(ids) is tuple and ids and all(type(v) is KernelId and
                v.group == self.group for v in ids) and len(set(ids)) == len(ids),
                'invalid backend kernel-domain expansion')
        return ids

    def collapse(self, ids, allocations, validator=None):
        require(type(ids) is tuple and all(type(v) is KernelId for v in ids),
                'collapse requires explicit kernel-domain IDs')
        require(type(allocations) is tuple and len(set(allocations)) == len(allocations),
                'unique allocation-domain candidates required')
        expected = tuple(k for ref in allocations for k in self.expand(ref, validator))
        require(ids == expected, 'kernel table is not exact ordered allocation expansion')
        return allocations


def native_expansion(geometry, ref):
    """Exact BlockTable.map_to_kernel_blocks arithmetic, once at the boundary."""
    return tuple(KernelId(geometry.group, ref.value*geometry.ratio+i)
                 for i in range(geometry.ratio))


def verify_native_row(table, row, allocation_ids, group):
    """Independently compare native expanded row against canonical pool IDs."""
    ratio=int(table.blocks_per_kv_block);kernel=int(table.block_size)
    require(ratio>0 and kernel>0,'invalid native BlockTable geometry')
    expected=tuple(b*ratio+i for b in allocation_ids for i in range(ratio))
    count=int(table.num_blocks_per_row[row])
    actual=tuple(int(v) for v in table.block_table.np[row,:count])
    require(count==len(expected) and actual==expected,
            'native kernel row is not exact allocation-ID expansion')
    return dict(group=group,allocation_tokens=ratio*kernel,kernel_tokens=kernel,
                ratio=ratio,allocation_ids=list(allocation_ids),kernel_ids=list(actual))
