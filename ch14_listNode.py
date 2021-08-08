# 由于链表的结构，很多链表问题可以用递归来处理
# 链表与数组不同，不能任意取节点的值，需要通过指针找到该节点
# 同理，需要找到链表的结尾，才知道链表的长度
# 技巧1: 尽量处理当前节点的下一个节点
# 技巧2：建立一个虚拟节点(dummy node)

from os import environb
from typing import List


class ListNode():
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    # 链表的基本操作
    # 206. Reverse Linked List(Easy)
    # 包括2种写法：递归 & 非递归
    # 1. 递归
    def reverseList_recursive(self, head):
        if not head or not head.next:
            return head

        res = self.reverseList_recuresive(head.next)
        head.next.next = head
        head.next = None

        return res

    # 2. 非递归
    def reverseList_none_recursive(self, head):
        prev = None
        while head:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev


    # 21. Merge Two Sorted Lists(Easy)
    # 给定2个增序的链表，试将其合并成一个增序的链表
    # 包括2种写法：递归 & 非递归
    # 1. 递归
    def mergeTwoLists_recursive(self, l1, l2):
        if not l2:
            return l1
        if not l1:
            return l2

        if l1.val > l2.val:
            l2.next = self.mergeTwoLists_recursive(l1, l2.next)
            return l2
        else:
            l1.next = self.mergeTwoLists_recursive(l1.next, l2)
            return l1

    # 2. 非递归
    def mergeTwoLists_none_recursive(self, l1, l2):
        dummy = ListNode(0)
        p = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            print(p.val)
            p = p.next

        if l1:
            p.next = l1
        if l2:
            p.next = l2
        print(p.val)

        return dummy.next


    # 24. Swap Nodes in Pairs(Meidum)
    # 交换节点
    # 交换2个节点，需要一个缓冲节点
    def swapPairs(self, head):
        dummy = ListNode(0)
        dummy.next = head
        tmp = dummy
        while tmp.next and tmp.next.next:
            node1 = tmp.next
            node2 = tmp.next
            tmp.next = node2
            node1.next = node2.next
            node2.next = node1
            tmp = node1
        return dummy.next


    # 其他链表技巧
    # 160. Intersection of Two Lists(Easy)
    def getIntersectionNode(headA, headB):
        l1 = headA
        l2 = headB
        while l1 != l2:
            if l1:
                l1 = l1.next
            else:
                l1 = headB

            if l2:
                l2 = l2.next
            else:
                l2 = headA

        return l1


    # 234. Palindrome Linked List(Easy)
    # 以 O(1) 的空间复杂度，判断链表是否回文
    # 1. 快慢指针，找到链表中点
    # 2. 翻转后半段链表
    # 3. 比较这两段链表
    def isPalindrome(self, head):
        if not head or not head.next:
            return True

        fast = head
        slow = head

        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        slow.next = self.reverseList(slow.next)
        slow = slow.next
        while slow:
            if head.val != slow.val:
                return False
            slow = slow.next
            head = head.next
        return True

    def reverseList(self, head):
        prev = None
        while head:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev


    # Basic Excises
    # 83. Remove Duplicates from Sorted List(Easy)
    def deleteDuplicates(self, head):
        p = head
        while p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next

        return head


    # 328. Odd Even linked List(Medium)
    # 空间复杂度O(1), 时间复杂度O(n)
    def oddEvenList(self, head):
        if not head:
            return head

        odd = head
        even = head.next
        even_head = even

        while even and even.next:
            odd.next = even.next
            odd = odd.next

            even.next = odd.next
            even = even.next

        odd.next = even_head
        return head


    # 19. Remove Nth Node from End of List(Medium)
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0, head)
        fast = slow = dummy

        for i in range(n):
            fast = fast.next

        while fast.next:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next
        return dummy.next


    # Harder Excises
    # 148. Sort List(Medium)
    # 利用快慢指针找到链表中点后，可以对链表进行归并排序
    def sortList(self, head):
        res = self.find_middle(head, None)
        return res

    def find_middle(self, head, tail):
        if not head:
            return head
        if head.next == tail:
            head.next = None
            return head

        slow = fast = head
        while fast != tail:
            slow = slow.next
            fast = fast.next
            if fast != tail:
                fast = fast.next

        mid = slow
        return self.merge(self.find_middle(head, mid), self.find_middle(mid, tail))

    def merge(self, left, right):
        dummy = ListNode(0)
        p = dummy
        l = left
        r = right

        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
            else:
                p.next = r
                r = r.next
            p = p.next

        if l:
            p.next = l
        if r:
            p.next = r
        return dummy.next





if __name__ == '__main__':
    s = Solution()
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(4)
    node1.next = node2
    node2.next = node3
    l1 = node1

    node4 = ListNode(1)
    node5 = ListNode(3)
    node6 = ListNode(4)
    node4.next = node5
    node5.next = node6
    l2 = node4

    res = s.mergeTwoLists_none_recursive(l1, l2)
    # while res:
    #     print(res.val)
    #     res = res.next
    print(res)
