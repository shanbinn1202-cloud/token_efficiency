from pathlib import Path
import difflib,json,hashlib
REV=Path(__file__).resolve().parent;D=REV.parents[1];P=D/'selective-prefix/source-overlay'
files=['vllm/v1/core/sched/scheduler.py','vllm/v1/core/kv_cache_manager.py']
s=(P/files[0]).read_text()
old='''            if self.defer_block_free:
                # Deferred frees hold blocks of several pools in one list.
                raise ValueError(
                    "decoupled KV layout does not support deferred block frees "
                    "(overlapping batches with a consumer KV connector)"
                )
'''
assert s.count(old)==1;s=s.replace(old,'')
s=s.replace('self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()','self.deferred_frees: deque[tuple[int, list[tuple[int, list[KVCacheBlock]]]]] = deque()')
s=s.replace('blocks = self.kv_cache_manager.pop_blocks_for_free(request)\n        if blocks:\n            self.deferred_frees.append((self.sched_step_seq, blocks))','blocks_by_pool = self.kv_cache_manager.pop_blocks_for_free_by_pool(request)\n        if blocks_by_pool:\n            self.deferred_frees.append((self.sched_step_seq, blocks_by_pool))')
s=s.replace('''            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))''','''            _, blocks_by_pool = self.deferred_frees.popleft()
            # Block IDs are pool-local. Keep each owning pool until the same
            # native in-flight-step fence has completed; never cross-free.
            for pool_id, blocks in blocks_by_pool:
                self.kv_cache_manager.block_pools[pool_id].free_blocks(reversed(blocks))''')
k=(P/files[1]).read_text();anchor='    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:\n';assert k.count(anchor)==1
new='''    def pop_blocks_for_free_by_pool(self, request: Request) -> list[tuple[int, list[KVCacheBlock]]]:
        """Detach bookkeeping without releasing in-flight pages in any pool.

        Multiple groups can share a pool; retain every reference, including
        repeated block objects, just as the native flattened shared-pool path.
        The scheduler releases each pool's references after its FIFO fence.
        """
        grouped: dict[int, list[KVCacheBlock]] = {}
        for pool_id, manager in zip(
            self.coordinator.group_pool, self.coordinator.single_type_managers,
            strict=True,
        ):
            blocks = manager.pop_blocks_for_free(request.request_id)
            if blocks:
                grouped.setdefault(pool_id, []).extend(blocks)
        return list(grouped.items())

'''
k=k.replace(anchor,new+anchor)
patch=[]
for name,after in zip(files,[s,k]):
 before=(P/name).read_text();o=REV/'original'/name;n=REV/'candidate'/name;o.parent.mkdir(parents=True,exist_ok=True);n.parent.mkdir(parents=True,exist_ok=True)
 assert not o.exists();o.write_text(before);n.write_text(after)
 compile(after,name,'exec')
 patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(REV/'changes.patch').write_text(''.join(patch))
(REV/'status.json').write_text(json.dumps({'status':'candidate_not_applied','reason':'GPU3447727 rejected async consumer connector with decoupled pools','algorithm_changes':False,'scheduler_fence_unchanged':True,'changes':'Preserve pool identity when detaching and releasing deferred KV pages; selector, ranking, turn ownership and early-free fence unchanged'},indent=2))
