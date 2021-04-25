class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    # Two Sum
    # 167. Two Sum II - Input array is sorted(Easy)
    def twoSum(self, numbers, target):
        left = 0
        right = len(numbers) - 1

        while left < right:
            tmp = numbers[left] + numbers[right]
            if tmp == target:
                return [left + 1, right + 1]

            elif tmp < target:
                left += 1

            else:
                right -= 1

        
    # 归并两个有序数组
    # 88. Merge Sorted Array(Easy)
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

        # 统计T中元素，及数量
        for c in t:
            if c in need:
                need[c] += 1
            else:
                need[c] = 1
        
        found = {}
        res = []

        l = 0
        for r in range(len(s)): 
            if s[r] in need:
                if s[r] in found:
                    found[s[r]] += 1
                else:
                    found[s[r]] = 1  

                while self.check(need, found):
                    if s[l] in need:
                        if found[s[l]] == need[s[l]]:
                            res.append(s[l: r+1])
                            found[s[l]] -= 1
                            l += 1
                            break
                        else:
                            found[s[l]] -= 1
                            l += 1    
                    else:
                        l += 1

        print(res)
                        
        if res == []:
            return ''
        else:
            length = float("inf")
            for r in res:
                if len(r) < length:
                    ans = r
                    length = len(r)
            return ans

    def check(self, need, found): 
        flag = 1
        for key, value in need.items():
            if key in found and found[key] >= value:
                pass
            else:
                flag = -1
                break
        return flag > 0


    # Basic Excises 
    # 633. Sum of Square Numbers(Medium)
    def judgeSquareSum(self, c):
        l = 0
        r = int(c ** 0.5)

        if l == r:
            return True
        
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
    def findLongestWord(self, s, dictionary):
        longest = 0
        pos = -1
        for i, word in enumerate(dictionary):
            p1 = len(s) - 1
            p2 = len(word) - 1

            while p1 >= 0 and p2 >= 0:
                if s[p1] == word[p2]:
                    p1 -= 1
                    p2 -= 1
                else:
                    p1 -= 1
            
            if p2 == -1:
                if len(word) > longest:
                    longest = len(word)
                    pos = i
                elif len(word) == longest:
                    if word < dictionary[pos]:
                        pos = i

        if pos != -1:
            return dictionary[pos]
        else:
            return ''

    
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
    # res = s.minWindow("cabwefgewcwaefgcf", "cae")
    # res = s.judgeSquareSum(1000000)
    # res = s.validPalindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga")
    # res = s.findLongestWord("abpcplea", ["a","b","c"])
    res = s.findLongestWord("abce", ["abe","abc"])

    print(res)

            
            