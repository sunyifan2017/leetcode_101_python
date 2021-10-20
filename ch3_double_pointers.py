"""
双指针主要用来遍历数组，2个指针指向不同的元素
(1) 2个指针指向同一个数组，遍历方向相同且不相交，称为滑动窗口
(2) 2个指针指向同一个数组，遍历方向相反，则用来搜索，
    待搜索的数组往往是拍好序的
"""

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    # Two Sum
    # 167. Two Sum II - Input array is sorted(Easy)
    # 两数之和，输入有序数组
    def twoSum(self, numbers, target):
        left = 0
        right = len(numbers) - 1

        while left < right:
            tmp = numbers[left] + numbers[right]
            if tmp == target:
                return [left  + 1, right + 1]

            elif tmp < target:
                left += 1

            else:
                right -= 1


    # 归并两个有序数组
    # 88. Merge Sorted Array(Easy)
    # 将第二个数组归并入第一个数组
    def merge(self, nums1, m, nums2, n):
        p1 = m - 1
        p2 = n - 1
        pos = m + n - 1

        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[pos] = nums1[p1]
                pos -= 1
                p1  -= 1
            else:
                nums1[pos] = nums2[p2]
                pos -= 1
                p2  -= 1

        if p2 >= 0:
            nums1[: pos+1] = nums2[: p2+1]

        return nums1


    # 快慢指针
    # 142. Linked List Cycle II(Medium)
    # 给定一个链表，如果有环路，找到环路的开始点
    def detectCycle(self, head):
        """
        3 -> 2 -> 0 -> 4
          a  ^    b    |
             | __   __ |
        """
        # 判断是否存在环路
        slow = head
        fast = head

        while True:
            if not fast.next or not fast:
                return None

            fast = fast.next.next
            slow = slow.next

            if fast ==  slow:
                break

        # 寻找环路节点
        # f = 2s, f = s + nb => s = nb, f = 2nb
        # 走到链表入口节点时的步数: k = a + nb => k = s + a
        # 重新计步
        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next

        return fast


    # 滑动窗口
    # 76. Minimum Window Substring(Hard)
    # Need to optim
    def minWindow(self, s, t):
        need = {}
        found = {}
        length = float('inf')
        res = ''
        l = 0

        for i in range(len(t)):
            need[t[i]] = need.get(t[i], 0) + 1

        for r in range(len(s)):
            if s[r] in need:
                found[s[r]] = found.get(s[r], 0) + 1
                while self.check(need, found):
                    if length > len(s[l: r+1]):
                        res = s[l: r+1]
                        length = len(res)
                    if s[l] in need:
                        found[s[l]] -= 1
                    l += 1

        if len(res) < length:
            return ""
        else:
            return res



    def check(self, need, found):
        for k, v in need.items():
            if k in found and found[k] >= v:
                pass
            else:
                return False
        return True


    # Basic Excises
    # 633. Sum of Square Numbers(Medium)
    # 平方数之和
    def judgeSquareSum(self, c):
        l = 0
        r = int(c ** 0.5)

        while l <= r:
            tmp = l * l + r * r
            if tmp > c:
                r -= 1
            elif tmp < c:
                l += 1
            else:
                return True

        return False


    # Basic Excises
    # 680. Valid palindrome II(Easy)
    # 验证回文字符串II
    # 给定一个非空字符串s，最多删除一个字符，判断能否成为回文字符串
    def validPalindrome(self, s):
        l = 0
        r = len(s) - 1

        res1 = True
        res2 = True
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                res1 = self.checkPalindrome(s[l: r])
                res2 = self.checkPalindrome(s[l+1: r+1])
                break

        return res1 or res2

    def checkPalindrome(self, s):
        l = 0
        r = len(s) - 1
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                return False
        return True


    # Basic Excises
    # 524. Longest Word in Dictionary through Deleting(Medium)
    # 通过删除字符匹配到字典里最长单词
    # 归并2个有序数组的变形题
    def findLongestWord(self, s, dictionary):
        max_len = 0
        ans = ''

        for word in dictionary:
            p1 = len(s) - 1
            p2 = len(word) - 1
            while p1 > -1 and p2 > -1:
                if s[p1] == word[p2]:
                    p1 -= 1
                    p2 -= 1
                else:
                    p1 -= 1

            if p2 == -1:
                if len(word) > max_len:
                    max_len = len(word)
                    ans = word
                elif len(word) == max_len:
                    if word < ans:
                        ans = word

        return ans


    # Harder Excises
    # 340. Longeset Substring with At Most K Distinct Characters(Hard)
    # 给一个字符串 s 和整数 k ， 要求返回最多由k个不同字符组成的最长子串
    def lengthOfLongestSubstringKDistinct(self, s, k):
        pass





if __name__ == '__main__':
    s = Solution()
    # res = s.twoSum([2, 7, 11, 15], 9)
    # res = s.merge([0], 0, [1], 1)
    # res = s.minWindow("ADOBECODEBANC", "ABC")
    # res = s.minWindow("bba", "ab")
    # res = s.judgeSquareSum(1000000)
    # res = s.validPalindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga")
    res = s.findLongestWord("abpcplea", ["a","b","c"])
    # res = s.findLongestWord("abce", ["abe","abc"])

    print(res)
