# -*- coding: utf-8 -*-

# 冒泡排序（Bubble Sort）
# 比较相邻的元素。如果第一个比第二个大，就交换它们两个；
# 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
# 针对所有的元素重复以上的步骤，除了最后一个；
# 重复步骤1~3，直到排序完成。


def bubble_sort(input):
    print("\nBubble Sort")
    input_len = len(input)
    print("length of input: %d" % input_len)
    for i in range(0, input_len):
        for j in range(0, input_len - 1 - i):
            if input[j] > input[j + 1]:
                tmp = input[j + 1]
                input[j + 1] = input[j]
                input[j] = tmp
    return input


test_arr = [3, 4, 1, 6, 30, 5]
test_arr_bubble_sorted = bubble_sort(test_arr)
print(test_arr_bubble_sorted)


# 选择排序(Selection-sort)
# 选择排序(Selection-sort)是一种简单直观的排序算法。它的工作原理：首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，
# 然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

def select_sort(input):
    print("\nSelect Sort")
    input_len = len(input)
    for i in range(0, input_len):
        min_index = i
        for j in range(i + 1, input_len):
            if input[j] < input[min_index]:
                min_index = j

        tmp = input[i]
        input[i] = input[min_index]
        input[min_index] = tmp
    return input


test_arr = [3, 4, 1, 6, 30, 5]
test_arr_select_sorted = select_sort(test_arr)
print(test_arr_select_sorted)

# 插入排序（Insertion Sort）
# 插入排序（Insertion-Sort）的算法描述是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，
# 在已排序序列中从后向前扫描，找到相应位置并插入。

# 归并排序（Merge Sort）
# 首先归并排序使用了二分法，归根到底的思想还是分而治之。拿到一个长数组，将其不停的分为左边和右边两份，然后以此递归分下去。
# 然后再将她们按照两个有序数组的样子合并起来。

import math


def merge_sort(input):
    input_len = len(input)
    if input_len <= 1:
        return input
    mid = math.floor(input_len / 2)
    left = merge_sort(input[:mid])
    right = merge_sort(input[mid:])
    return merge(left, right)


def merge(sorted_arr1, sorted_arr2):
    result = []
    i = j = 0
    while i < len(sorted_arr1) and j < len(sorted_arr2):
        if sorted_arr1[i] < sorted_arr2[j]:
            result.append(sorted_arr1[i])
            i = i + 1
        else:
            result.append(sorted_arr2[j])
            j = j + 1

    if i == len(sorted_arr1):
        for item in sorted_arr2[j:]:
            result.append(item)
    else:
        for item in sorted_arr1[i:]:
            result.append(item)
    return result


test_arr = [3, 4, 1, 6, 30, 5]
print("\nMerge Sort")
test_arr_merge_sorted = merge_sort(test_arr)
print(test_arr_merge_sorted)


# 快速排序（Quick Sort）
# 快速排序使用分治法来把一个串（list）分为两个子串（sub-lists）。具体算法描述如下：
#
# 从数列中挑出一个元素，称为 “基准”（pivot）；
# 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
# 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

# def quick_sort(li, start, end):
#     # 分治 一分为二
#     # start=end ,证明要处理的数据只有一个
#     # start>end ,证明右边没有数据
#     if start >= end:
#         return
#     # 定义两个游标，分别指向0和末尾位置
#     left = start
#     right = end
#     # 把0位置的数据，认为是中间值
#     mid = li[left]
#     while left < right:
#         # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
#         while left < right and li[right] >= mid:
#             right -= 1
#         li[left] = li[right]
#         # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
#         while left < right and li[left] < mid:
#             left += 1
#         li[right] = li[left]
#     # while结束后，把mid放到中间位置，left=right
#     li[left] = mid
#     # 递归处理左边的数据
#     quick_sort(li, start, left - 1)
#     # 递归处理右边的数据
#     quick_sort(li, left + 1, end)


def quick_sort(array, l, r):
    if l < r:
        q = partition(array, l, r)
        quick_sort(array, l, q - 1)
        quick_sort(array, q + 1, r)


def partition(array, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1
