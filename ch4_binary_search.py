"""
二分查找：
对于一个长度为n的数组，二分查找的时间复杂度为O(log n)

涂色方法模版：
https://www.bilibili.com/video/BV1d54y1q7k7?from=search&seid=9396313407709460775

l = -1, r = N （必须）
while l+1 != r:
    m = l + (r - l) // 2
    if IsBLUE(m):
        l = m
    else:
        r = m
return l or r
"""


class Solution(object):
    # 求开方
    # 69. Sqrt(x) (Easy)
    def mySqrt(self, x):
        if x < 2:
            return x

        l = 0
        r = x
        while l <= r:
            mid = l + (r - l) // 2
            sqrt = x // mid

            if sqrt == mid:
                return mid
            elif sqrt > mid:
                l = mid + 1
            else:
                r = mid - 1

        return r

    def mySqrt_template(self, x):
        # [0, 1, 2, ..., x], isBLUE(m)= x // m <= m, return l
        if x < 2:
            return x

        l = -1
        r = x + 1
        while l + 1 != r:
            m = l + (r - l) // 2
            isBLUE = x // m >= m
            if isBLUE:
                l = m
            else:
                r = m

        return l

    # 牛顿迭代法
    # x^2 = a
    # f(x) = x^2 - a = 0的解
    # f'(x) = 2*x
    # x_{n+1} = (x_n + a / x+n) / 2
    def mySqrt_Newton(self, a):
        x = a
        while x*x > a:
            x = (x + a / x) // 2

        return int(x)


    # 查找区间
    # 34. Find First and Last Position of Element in Sorted Array(Medium)
    def searchRange(self, nums, target):
        if not nums:
            return [-1, -1]

        l = self.lower_bound(nums, target)
        r = self.upper_bound(nums, target)

        return [l, r]

    def lower_bound(self, nums, target):
        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                r = mid - 1
            elif nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1

        # 判断边界
        if l >= len(nums) or nums[l] != target:
            return -1
        return l

    def upper_bound(self, nums, target):
        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1

        # 判断边界
        if r < 0 or nums[r] != target:
            return -1
        return r


    def searchRange_template(self, nums, target):
        if not nums:
            return [-1, -1]

        l = self.lower_bound_template(nums, target)
        r = self.upper_bound_template(nums, target)

        return [l, r]

    def lower_bound_template(self, nums, target):
        # isBLUE(m) = nums[m] < target, return l
        l = -1
        r = len(nums)

        while l + 1 != r:
            m = l + (r - l) // 2
            isBLUE = nums[m] < target
            if isBLUE:
                l = m
            else:
                r = m

        if r == len(nums) or nums[r] != target:
            return -1

        return r

    def upper_bound_template(self, nums, target):
        # isBLUE(m) = nums[m] <= target, return l
        l = -1
        r = len(nums)

        while l + 1 != r:
            m = l + (r - l) // 2
            isBLUE = nums[m] <= target
            if isBLUE:
                l = m
            else:
                r = m

        if l != -1 and nums[l] != target:
            return -1
        return l


    # 旋转数组查找数字
    # 81. Search in Rotated Sorted Array II(Medium)
    # 旋转数组，涂色方法失效
    # 找到保证递增的区间
    def search(self, nums, target):
        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return True

            if nums[l] == nums[mid] and nums[mid] == nums[r]:
                l += 1
                r -= 1

            # 左侧递增
            elif nums[l] <= nums[mid]:
                if nums[l] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1

            # 右侧递增
            else:
                if nums[mid] < target and target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1

        return False

    def search_template(self, nums, target):
        # isBLUE(m)= nums[m] <= target, return l,
        # patch l != -1 and nums[l] != target, return -1

        pass


    # Basic Excises
    # 154. Find Minimum in Rotated Sorted Array II(Hard)
    # 寻找旋转排序数组中的最小数
    def findMin(self, nums):
        """
        a[0], a[1], a[2], ..., a[n-1] ->翻转一次-> a[n-1], a[0], a[1], ...
        相当与寻找旋转断点.
        """
        l = 0
        r = len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2

            if nums[l] == nums[mid] and nums[mid] == nums[r]:
                l += 1
                r -= 1

            # 左侧递增
            elif nums[l] <= nums[mid]:
                # 无旋转
                if nums[l] < nums[r]:
                    return nums[l]
                # 断点在右侧
                else:
                    l = mid + 1

            #  右侧递增
            else:
                # 无旋转
                if nums[l] < nums[r]:
                    return nums[l]
                # 断点在左侧
                else:
                    r = mid

        return nums[l]


    # Basic Excises
    # 540. Single Element in a Sorted Array(Medium)
    def singleNonDuplicate(self, nums):
        """
        对偶数进行索引，直到遇到第一个其后元素不相同的索引.
        """
        l = 0
        r = len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] == nums[mid+1]:
                l = mid + 2
            else:
                r = mid

        return nums[l]


    # Harder Excises
    # 4. Median of Two Sorted Arrays(Hard)
    def findMedianSortedArrays(self, nums1, nums2):
        pass



if __name__ == '__main__':
    s = Solution()
    # res = s.mySqrt(3)
    # res = s.mySqrt_template(0)
    res = s.mySqrt_Newton(0)
    # res = s.searchRange([5, 7, 7, 8, 8, 10], 8)
    # res = s.searchRange_template([1, 2], 1)
    # res = s.search([2, 5, 6, 0, 0, 1, 2], 3)
    # res = s.search([5, 1, 3], 3)
    # res = s.findMin([3, 3, 1, 3])
    # res = s.singleNonDuplicate([2,2,3])

    print(res)