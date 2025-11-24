---
title: "Iterator in python"
subtitle: "Learning Hugo + LoveIt"
date: '2025-11-24T19:34:57+08:00'
draft: false
tags: ["hugo", "loveit", "learning"]
categories: ["blog"]
author: "Your Name"

summary: "This is my first post on my personal site using Hugo LoveIt theme. Exploring markdown, code blocks, images, and shortcodes."
---

## insert_items

- 在当前元素后面插入元素后，改变了`len(s)`，故在迭代时不可写`for i in range(len(s))`这样限制迭代次数的语句；并且在迭代时，主动跳过插入的元素，防止`large_s = [1, 4, 8]`反复插入4

- 后序遍历由于跳过了插入的元素，更简洁

> `list.insert(i, x)`
> Insert an item at a given position. The first argument is the index of the element before which to insert, so `a.insert(0, x)` inserts at the front of the list, and `a.insert(len(a), x)` is equivalent to `a.append(x)`.


```python
def insert_items(s: list[int], before: int, after: int) -> list[int]:
    """Insert after into s following each occurrence of before and then return s.

    >>> test_s = [1, 5, 8, 5, 2, 3]
    >>> new_s = insert_items(test_s, 5, 7)
    >>> new_s
    [1, 5, 7, 8, 5, 7, 2, 3]
    >>> test_s
    [1, 5, 7, 8, 5, 7, 2, 3]
    >>> new_s is test_s
    True
    >>> double_s = [1, 2, 1, 2, 3, 3]
    >>> double_s = insert_items(double_s, 3, 4)
    >>> double_s
    [1, 2, 1, 2, 3, 4, 3, 4]
    >>> large_s = [1, 4, 8]
    >>> large_s2 = insert_items(large_s, 4, 4)
    >>> large_s2
    [1, 4, 4, 8]
    >>> large_s3 = insert_items(large_s2, 4, 6)
    >>> large_s3
    [1, 4, 6, 4, 6, 8]
    >>> large_s3 is large_s
    True
    """
    index = 0
    while index < len(s):
        if s[index] == before:
            s.insert(index + 1, after)
            index += 1
        index += 1
    return s

    # Alternate solution (backward traversal):
    i = len(s) - 1
    while i >= 0:
        if s[i] == before:
            s.insert(i + 1, after)
        i -= 1
    return s
```

## repeated

```python
from typing import Iterator  # "t: Iterator[int]" means t is an iterator that yields integers

def repeated(t: Iterator[int], k: int) -> int:
    """Return the first value in iterator t that appears k times in a row,
    calling next on t as few times as possible.

    >>> s = iter([10, 9, 10, 9, 9, 10, 8, 8, 8, 7])
    >>> repeated(s, 2)
    9
    >>> t = iter([10, 9, 10, 9, 9, 10, 8, 8, 8, 7])
    >>> repeated(t, 3)
    8
    >>> u = iter([3, 2, 2, 2, 1, 2, 1, 4, 4, 5, 5, 5])
    >>> repeated(u, 3)
    2
    >>> repeated(u, 3)
    5
    >>> v = iter([4, 1, 6, 6, 7, 7, 8, 8, 2, 2, 2, 5])
    >>> repeated(v, 3)
    2
    """
    assert k > 1
    count = 0
    last_item = None # 更贴近内涵
    while True:
        item = next(t)
        if item == last_item:
            count += 1
        else:
            last_item = item
            count = 1
        if count == k:
            return item

    # 我的写法
    assert k > 1
    count = 1
    pre = next(t) # 优先迭代一次
    try:
        while True:
            cur = next(t)
            if cur == pre:
                count += 1
            else:
                pre = cur
                count = 1
            if count == k:
                return cur
    except StopIteration:
        print('No value in t that appears k times in a row!')
        return None
```

## partial_reverse

python里面`a, b, c, ... = ..., c, b, a`语句是给等号右边创建一个元组,然后解包后赋值给等号左边的元素.

```python
def partial_reverse(s: list[int], start: int) -> None:
    """Reverse part of a list in-place, starting with start up to the end of
    the list.

    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> partial_reverse(a, 2)
    >>> a
    [1, 2, 7, 6, 5, 4, 3]
    >>> partial_reverse(a, 5)
    >>> a
    [1, 2, 7, 6, 5, 3, 4]
    """
    end = len(s) - 1
    while start < end:
        s[start], s[end] = s[end], s[start]
        start, end = start + 1, end - 1
```

拓展**旋转数组**算法,要求原地反转,空间复杂度$O(1)$

1. 反转整个数组
2. 反转前k个元素
3. 反转后n-k个元素

```python
def rotate_array(nums: list[int], k: int) -> None:
    """将数组向右旋转k步
    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> rotate_array(a, 3)
    >>> a
    [5, 6, 7, 1, 2, 3, 4]
    >>> rotate_array(a, 10)
    >>> a
    [5, 6, 7, 1, 2, 3, 4]
    """
    n = len(nums)
    k %= n  # 处理k > n的情况

    # 定义类似上题的函数作为辅助
    def reverse(start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start + 1, end - 1

    reverse(0, n - 1)   # 反转整个数组
    reverse(0, k - 1)   # 反转前k个元素
    reverse(k, n - 1)   # 反转后n-k个元素
```