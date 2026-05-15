# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm import envs
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    SHORTEST = "shortest"
    SHORTEST_AGING = "shortest_aging"


def shortest_request_key(request: Request) -> tuple[int, float, str, int]:
    # 短请求优先策略只改变等待队列顺序，不改变 token budget
    # 或 KV cache 分配逻辑；相同长度时保持先到先服务，避免结果不稳定。
    return (
        request.num_prompt_tokens,
        request.arrival_time,
        request.request_id,
        id(request),
    )


def shortest_aging_request_key(
    request: Request,
    now: float | None = None,
    aging_weight: float | None = None,
) -> tuple[float, float, int, str, int]:
    # Aging 策略的核心：短 prompt 仍然优先，但等待时间会抵消一部分
    # prompt 长度。这样长请求等待足够久后也能逐渐获得调度机会。
    if now is None:
        now = time.time()
    if aging_weight is None:
        aging_weight = envs.VLLM_SHORTEST_AGING_WEIGHT
    waited_s = max(0.0, now - request.arrival_time)
    effective_prompt_tokens = max(
        0.0,
        request.num_prompt_tokens - waited_s * aging_weight,
    )
    return (
        effective_prompt_tokens,
        request.arrival_time,
        request.num_prompt_tokens,
        request.request_id,
        id(request),
    )


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class ShortestRequestQueue(RequestQueue):
    """
    A shortest-prompt-first queue that supports heap operations.

    Requests with fewer prompt tokens are processed first. If multiple requests
    have the same prompt length, the one with the earlier arrival time is
    processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[tuple[int, float, str, int], Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request according to shortest-prompt-first policy."""
        heapq.heappush(self._heap, (shortest_request_key(request), request))

    def pop_request(self) -> Request:
        """Pop the shortest prompt request."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the shortest prompt request without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0][1]

    def prepend_request(self, request: Request) -> None:
        """Add a request according to shortest-prompt-first policy.

        最短请求优先队列没有真正的“插到队头”语义
        被重新放回来的请求仍按 prompt 长度和到达时间排序。
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests according to shortest-prompt-first policy."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(key, req) for key, req in self._heap if req is not request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [
            (key, req) for key, req in self._heap if req not in requests_to_remove
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to shortest-prompt-first policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, request = heapq.heappop(heap_copy)
            yield request


class ShortestAgingRequestQueue(RequestQueue):
    """
    A shortest-prompt-first queue with waiting-time aging.

    Aging priority changes over time, so this queue intentionally uses a
    linear scan instead of a heap with stale keys.
    """

    def __init__(self) -> None:
        self._requests: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request according to shortest-aging policy."""
        self._requests.append(request)

    def _best_index(self) -> int:
        if not self._requests:
            raise IndexError("select from empty queue")
        now = time.time()
        aging_weight = envs.VLLM_SHORTEST_AGING_WEIGHT
        # 中文注释：aging 的 key 会随时间变化，不能像 shortest 那样
        # 入队时固定 heap key；每次选择时重新计算，保证等待补偿生效。
        best_idx = 0
        best_key = shortest_aging_request_key(
            self._requests[0], now, aging_weight
        )
        for idx, request in enumerate(self._requests[1:], start=1):
            key = shortest_aging_request_key(request, now, aging_weight)
            if key < best_key:
                best_idx = idx
                best_key = key
        return best_idx

    def pop_request(self) -> Request:
        """Pop the best request under shortest-aging policy."""
        return self._requests.pop(self._best_index())

    def peek_request(self) -> Request:
        """Peek at the best request under shortest-aging policy."""
        return self._requests[self._best_index()]

    def prepend_request(self, request: Request) -> None:
        """Add a request according to shortest-aging policy.

        shortest_aging 没有真正的“插队头”语义；重新入队的请求
        会在下一次选择时按当前等待时间重新计算优先级。
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests according to shortest-aging policy."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._requests.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._requests = [
            req for req in self._requests if req not in requests_to_remove
        ]

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._requests)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to current shortest-aging policy."""
        requests = self._requests[:]
        now = time.time()
        aging_weight = envs.VLLM_SHORTEST_AGING_WEIGHT
        while requests:
            best_idx = min(
                range(len(requests)),
                key=lambda idx: shortest_aging_request_key(
                    requests[idx], now, aging_weight
                ),
            )
            yield requests.pop(best_idx)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.SHORTEST:
        return ShortestRequestQueue()
    elif policy == SchedulingPolicy.SHORTEST_AGING:
        return ShortestAgingRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
