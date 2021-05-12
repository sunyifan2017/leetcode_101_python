class Solution(object):
    # 选择排序
    # 需要新的内存空间
    # 时间复杂度 O(n^2)
    def selection_sort(self, nums):
        new = []
        for _ in range(len(nums)):
            smallest_idx = self.find_smallest(nums)
            new.append(nums.pop(smallest_idx))
        return new

    def find_smallest(self, nums):
        smallest = nums[0]
        idx = 0
        for i in range(1, len(nums)):
            if nums[i] < smallest:
                smallest = nums[i]
                idx = i
        return idx

    def selection_sort_simple(self, nums):
        n = len(nums)
        # 外层循环，表示循环选择的遍数, 被交换位置
        for i in range(n-1):
            # 内层循环，找到最小的pos
            min_index = i
            for j in range(i+1, n):
                if nums[j] < nums[min_index]:
                    min_index = j
            nums[i], nums[min_index] = nums[min_index], nums[i]


    # 快速排序
    # O(nlogn)
    # 分而治之策略(递归)
    def quick_sort_On(self, nums):
        # 基线条件
        if len(nums) < 2:
            return nums

        # 分解问题
        # 1. 选择基准值；2. 将数组分为大于基准和小于基准的元素
        # 3. 对两个子数组进行快速排序
        else:
            pivot = nums[0]
            less = [num for num in nums[1:] if num <= pivot]
            greater = [num for num in nums[1:] if num > pivot]

        return self.quick_sort_On(less) + [pivot] + self.quick_sort_On(greater)


    def quick_sort(self, nums, l, r):
        if l + 1 >= r:
            return

        first = l
        last = r - 1
        pivot = nums[l]

        while first < last:
            while first < last and nums[last] >= pivot:
                last -= 1
            nums[first] = nums[last]
            while first < last and nums[first] <= pivot:
                first += 1
            nums[last] = nums[first]
        nums[first] = pivot

        self.quick_sort(nums, l, first)
        self.quick_sort(nums, first+1, r)


    # 冒泡排序
    def bubble_sort(self, nums):
        n = len(nums)

        # 第i轮比较n-i-i次
        for i in range(n):
            swapped = False
            # 内存循环：比较大小，互换位置
            for j in range(n-i-1):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
                    swapped = True

            if not swapped:
                break

        return nums


    # 插入排序
    # 稳定
    def insertion_sort(self, nums):
        n = len(nums)
        for i in range(n):
            j = i
            while j > 0 and nums[j] < nums[j-1]:
                nums[j], nums[j-1] = nums[j-1], nums[j]
                j -= 1

        return nums


    # 归并排序 
    # 合并2个有序数组 
    # 递归 O(nlogn)
    def merge_sort(self, nums):
        if len(nums) < 2:
            return nums

        # 二分
        mid = len(nums) // 2
        left = self.merge_sort(nums[: mid])
        right = self.merge_sort(nums[mid:])

        # 合并
        merged = []

        while left and right:
            if left[0] <= right[0]:
                merged.append(left.pop(0))
            else:
                merged.append(right.pop(0))
        merged.extend(right if right else left)

        return merged


    # 快速选择
    # 215. Kth Largest Element in an array
    # 与快速排序相似
    def findKthLargest(self, nums, k):
        # 时间复杂度O(n)，空间复杂度O(1)
        l = 0
        r = len(nums) - 1
        target = k - 1

        while l < r:
            mid = self.quick_seletion(nums, l, r)
            if mid == target:
                return nums[mid]
            elif mid > target:
                r = mid - 1
            else:
                l = mid + 1

        return nums[l]

    def quick_seletion(self, nums, l, r):
        # 无递归, 只分类
        first = l
        last = r
        pivot = nums[l]

        while first < last:
            while first < last and nums[last] <= pivot:
                last -= 1
            nums[first] = nums[last]
            while first < last and nums[first] >= pivot:
                first += 1
            nums[last] = nums[first]
        nums[first] = pivot

        return first


    # 桶排序：无比较
    # 计数排序
    # 量大取值范围小
    # 空间复杂度：n + k
    # 时间复杂度：n + k
    # 347.Top K Frequent Elements (Medium)
    def topKFrequent(self, nums, k):
        bucket = {}
        for num in nums:
            bucket[num] = bucket.setdefault(num, 0) + 1

        bucket_new = {}
        for key, value in bucket.items():
            if value in bucket_new:
                bucket_new[value].append(key)
            else:
                bucket_new[value] = [key]

        idx = sorted(list(bucket_new.keys()), key=lambda x: -x)
        i, count = 0, 0
        res = []
        while count < k and count < len(idx):
            for ans in bucket_new[idx[count]]:
                res.append(ans)
                i += 1

                if i >= k:
                    count = k
                    break
            count += 1

        return res

    
    # Basic Excises
    # 451. Sort Characters By Frequency(Medium)
    def frequencySort(self, s):
         pass


    # Harder Exciese
    # 75. Sort Colors(Medium)
    def sortColors(self, nums):
        n = len(nums)
        if n < 2:
            return

        # 快排思想
        # [0,　zero), [zero, i), [two, n-1]
        # 在遍历的过程中，「下标先加减再交换」
        # 还是「先交换再加减」就看初始化的时候变量在哪里
        zero = 0
        two = n
        i = 0

        while i < two:
            if nums[i] == 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                i += 1
                zero += 1
            elif nums[i] == 1:
                i += 1
            elif nums[i] == 2:
                two -= 1
                nums[i], nums[two] = nums[two], nums[i]



if __name__ == '__main__':
    s = Solution()
    # res = s.selection_sort([5, 1, 2, 8])
    # nums = [5, 4, 7, 2, 9, 1, 77]
    # s.selection_sort_simple(nums)
    # print(nums)

    # res = s.quick_sort_On([1, 2, 8, 5, 10, 16])

    # nums = [5, 4, 7, 2, 9, 1, 77]
    # s.quick_sort(nums, 0, len(nums))
    # print(nums)

    # res = s.bubble_sort([5, 1, 2, 8])
    # res = s.insertion_sort([5, 1, 2, 8])
    # res = s.merge_sort([7, 4, 9, 2, 1, 10])
    # res = s.findKthLargest([3, 2, 1, 5, 6, 4], 2)
    # res = s.topKFrequent([4,1,-1,2,-1,2,3], 2)

    nums = [2, 0, 2, 1, 1]
    s.sortColors(nums)
    print(nums)


    # print(res)

