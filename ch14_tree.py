from collections import deque
from typing import OrderedDict



# 树与链表的主要差别在于多了一个节点的指针
class TreeNode:
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


# 二叉树的遍历
class TreeTraversal:
    def preorder_recursion(self, root):
        if not root:
            return
        print(root.val)
        self.preorder_recursion(root.left)
        self.preorder_recursion(root.right)

    def preorder_stack(self, root):
        if not root:
            return

        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node:
                print(node.val)
                stack.append(node.left)
                stack.append(node.right)

    def inorder_recursion(self, root):
        if not root:
            return

        self.inorder_recursion(root.left)
        print(root.val)
        self.inorder_recursion(root.right)

    def inorder_stack(self, root):
        if not root:
            return

        stack = []
        while stack or root:
            while root:                     # 从根节点开始，一直找到ta的左子树
                stack.append(root)
                root = root.left
            root = stack.pop()              # while结束，当前节点无左子树
            print(root.val)
            root = root.right               # 查看右子树


    def postorder_recursion(self, root):
        if not root:
            return

        self.postorder_recursion(root.left)
        self.postorder_recursion(root.right)
        print(root.val)

    def postorder_stack(self, root):
        if not root:
            return

        stack1 = []
        stack2 = []
        stack1.append(root)
        while stack1:
            node = stack1.pop()
            if node.left:
                stack1.append(node.left)
            if node.right:
                stack1.append(node.right)
            stack2.append(node)

        while stack2:
            print(stack2.pop().val)


class Solution:
    # 树的递归
    # 与深度优先搜索的递归写法相同
    # 104. Maximum Depth of Binary Tree(Easy)
    def maxDepth(self, root: TreeNode) -> int:
        if root:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        else:
            return 0


    # 110. Balanced Binary Tree(Easy)
    # 判断一棵二叉树是否平衡
    # 树平衡：对于树上的任意节点，其两侧的节点的最大深度的差值不得大于1
    def isBalanced(self, root: TreeNode) -> bool:
        if self.helper(root) > -1:
            return True
        else:
            return False

    def helper_balanced(self, root: TreeNode) -> int:
        if not root:
            return 0

        left = self.helper_balanced(self, root.left)
        right = self.helper_balanced(self, root.right)

        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1

        return 1 + max(left, right)


    # 543. Diameter of Binary Tree(Easy)
    # 求一个二叉树的最长直径
    # 直径：二叉树上任意两节点之间的无向距离
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.diameter = 0
        self.helper_diameter(root)
        return self.diameter

    def helper_diameter(self, node: TreeNode) -> int:
        if not node:
            return 0

        left = self.helper_diameter(node.left)
        right = self.helper_diameter(node.right)

        self.diameter = max(left + right, self.diameter)              # 更新最长直径
        return 1 + max(left, right)                                   # 返回以该子树根节点为断点的最长直径值


    # 437. Path Sum III(Medium)
    # 给定一个整数二叉树，输出一个整数，表示有多少条满足条件的路径
    def pathSum(self, root, targetSum):
        # 递归每一个节点
        if not root:
            return 0

        mid = self.pathSumWithRoot(root, targetSum)
        left = self.pathSum(root.left, targetSum)
        right = self.pathSum(root.right, targetSum)

        return mid + left + right

    def pathSumWithRoot(self, root, targetSum):
        if not root:
            return 0

        # 当前节点是否满足条件
        if root.val == targetSum:
            count = 1
        else:
            count = 0

        count += self.pathSumWithRoot(root.left, targetSum-root.val)
        count += self.pathSumWithRoot(root.right, targetSum-root.val)

        return count


    # 101. Symmetric Tree(Easy)
    # 判断一个树是否对称
    # 四步法：
    # 1. 如果两个子树都为空，则他们相等OR对称
    # 2. 如果两个子树只有一个为空，则他们不相等OR不对称
    # 3. 如果两个子树的根节点的值不相等，则他们不相等OR不对称
    # 4. 根据相等或对称要求，进行递归处理
    def isSymmetric(self, root):
        if not root:
            return True

        return self.helper_symmetric(root.left, root.right)

    def helper_symmetric(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        return self.helper_symmetric(left.left, right.right) and \
            self.helper_symmetric(left.right, right.left)


    # 1110. Delete Nodes and Return Forest(Medium)
    # 给定一个整数二叉树和一些整数，求删掉这些整数对应的节点后，剩余的子树
    def delNodes(self, root, to_delete):
        forest = []
        record = {}     # 快速查找欲删除节点的数字
        for ele in to_delete:
            record[ele] = 1
        root = self.helper_del(root, record, forest)    # 无删除，返回元树
        if root:
            forest.append(root)
        return forest

    def helper_del(self, root, record, forest):
        if not root:
            return root

        root.left = self.helper_del(root.left, record, forest)      # 遍历左子树，返回左子树
        root.right = self.helper_del(root.right, record, forest)    # 遍历右子树，返回右子树

        if record.get(root.val, 0):                  # 如果满足删除条件
            if root.left:                            # 左子树为一棵新的独立的树
                forest.append(root.left)
            if root.right:                           # 右子树为一棵新的独立的树
                forest.append(root.right)
            root = None                              # 当前节点为None

        return root


    # 层次遍历
    # 广度优先搜索进行层次遍历
    # 637. Average of Level in Binary Tree(Easy)
    # 给定一个二叉树，求每一层的平均数
    def averageOflevels(self, root: TreeNode) -> list:
        ans = []
        if not root:
            return ans

        queue = deque()
        queue.append(root)
        while queue:
            n_nodes = len(queue)
            tmp = 0
            for i in range(n_nodes):
                node = queue.popleft()
                tmp += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(tmp / n_nodes)

        return ans


    # 前中后序遍历
    # * 深度优先搜索
    # 前序：根左右
    # 中序：左根右
    # 后序：左右根
    # 105. Construct Binary Tree from Preorder and Inorder Traversal(Medium)
    # 根据前序，中序还原树，节点无重复值
    def buildTree(self, preorder: list, inorder: list) -> TreeNode:
        if not preorder:
            return None

        hash = {}
        for i in range(len(inorder)):
            hash[inorder[i]] = i

        return self.helper_build_tree(preorder, hash, 0, 0, len(preorder)-1)

    def helper_build_tree(self, preorder, hash, start_pre, start_in, end_in):
        if start_in > end_in:
            return None

        root = preorder[start_pre]
        idx = hash[root]
        left_len = idx - start_in
        node = TreeNode(root)
        node.left = self.helper_build_tree(preorder, hash, start_pre+1, start_in, idx-1)
        node.right = self.helper_build_tree(preorder, hash, start_pre+left_len+1, end_in)

        return node


    # 144. Binary Tree Preorder Traversal(Medium)
    # 不使用递归，实现二叉树的前序遍历
    def preorderTraversal(self, root):
        ans = []
        if not root:
            return ans

        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            ans.append(node.val)
            if node.right:                  # 先右后左，因为栈先进后出
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return ans

    def preorderTravsersal_recursive(self, root):
        if not root:
            return []

        ans = [root.val]
        ans += self.preorderTravsersal_recursive(root.left)
        ans += self.preorderTravsersal_recursive(root.right)

        return ans


    # 二叉搜索树
    # 特殊的二叉树，左子树 < 父节点， 右子树 > 父节点
    # 可以在O(logn), 查找一个值是否存在，同二分查找
    # 二叉搜索树的中序遍历结果为排好序的数组

    # 99. Recover Binary Search Tree(Hard)
    # 给定一个二叉搜索树，已知有2个节点被不小心交换了，试复原此树
    # 中序遍历
    # 设置prev，使当前节点与prev比较，如果prev > node_i
    # 则当前节点错误
    # 遍历后，只出现一次错误，则node_i与prev错位，交换
    # 若出现二次错误，则交换这两个错误nodes
    def recoverTree(self, root):
        self.mistake1 = None
        self.mistake2 = None
        self.prev = None

        self.inorder(root)

        if self.mistake1 and self.mistake2:
            tmp = self.mistake1.val
            self.mistake1.val = self.mistake2.val
            self.mistake2.val = tmp

    def inorder(self, root):
        if not root:
            return

        self.inorder(root.left)

        if self.prev and self.prev.val > root.val:
            if not self.mistake1:
                self.mistake1 = self.prev
                self.mistake2 = root
            else:
                self.mistake2 = root
        self.prev = root

        self.inorder(root.right)


    # 669. Trim A Binary Tree(Medium)
    # 给定一个二叉查找树和两个整数 L 和 R，且 L < R，试修剪此二叉查找树，使得修剪后所有
    # 节点的值都在 [L, R] 的范围内
    def trimBST(self, root, low, high):
        if not root:
            return root
        if root.val > high:
            return self.trimBST(root.left, low, high)
        if root.val < low:
            return self.trimBST(root.right, low, high)

        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)

        return root


    # Basic Excises
    # 226. Invert Binary Tree(Easy)
    # 翻转二叉树
    def convertTree(self, root):
        if not root:
            return root

        left = self.convertTree(root.left)
        right = self.convertTree(root.right)

        root.left = right
        root.right = left

        return root


    # 617. Merge Two Binary Trees(Easy)
    # 合并二叉树
    # 1. 递归
    def mergeTrees(self, root1, root2):
        if not root1:
            return root2
        if not root2:
            return root1

        left = self.mergeTrees(root1.left, root2.left)
        right = self.mergeTrees(root1.right, root2.right)

        root1.val = root1.val + root2.val
        root1.left = left
        root1.right = right

        return root1

    # 2. 广度优先搜索
    def mergeTrees_bfs(self, root1, root2):
        if not root1:
            return root2
        if not root2:
            return root1

        merged = TreeNode(root1.val + root2.val)
        queue_m = deque([merged])
        queue_1 = deque([root1])
        queue_2 = deque([root2])
        while queue_1 and queue_2:
            node_m = queue_m.popleft()
            node_1 = queue_1.popleft()
            node_2 = queue_2.popleft()

            left_1, right_1 = node_1.left, node_1.right
            left_2, right_2 = node_2.left, node_2.right

            if left_1 or left_2:
                if left_1 and left_2:
                    left_m = TreeNode(left_1.val + left_2.val)
                    node_m.left = left_m
                    queue_m.append(left_m)
                    queue_1.append(left_1)
                    queue_2.append(left_2)
                elif left_1:
                    node_m.left = left_1
                else:
                    node_m.left = left_2

            if right_1 or right_2:
                if right_1 and right_2:
                    right_m = TreeNode(right_1.val + right_2.val)
                    node_m.right = right_m
                    queue_m.append(right_m)
                    queue_1.append(right_1)
                    queue_2.append(right_2)
                elif right_1:
                    node_m.right = right_1
                else:
                    node_m.right = right_2

        return merged


    # 572. Subtree of Another Tree(Easy)
    # 另一个树的子树
    def isSubtree(self, root, subRoot):
        # 遍历root
        if self.isSubtree(root, subRoot):
            return True
        if root:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        return False

    def isSameTree(self, s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        if s.val != t.val:
            return False

        return self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)


    # 404. Sum of Left Leaves(Easy)
    # 左叶子之和
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if not root:
            return 0

        queue = deque()
        queue.append(root)
        res = 0
        while queue:
            n_nodes = len(queue)
            for i in range(n_nodes):
                node = queue.pop()
                if node.left and not node.left.left and not node.left.right:
                    res += node.left.val
                elif node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res


    # 513. Find Bottom Left Tree Value(Medium)
    # 找树左下角的值
    def findBottomLeftValue(self, root):
        if not root:
            return 0

        queue = deque()
        queue.append(root)
        while queue:
            level = []
            n_nodes = len(queue)
            for i in range(n_nodes):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return level[0]


    # 538. Convert BST to Greater Tree(Medium)
    # 把二叉搜索树转换为累加树
    def convertBST(self, root):
        if not root:
            return

        stack = []
        greater = 0
        head = root
        while stack or root:
            while root:
                stack.append(root)
                root = root.right
            root = stack.pop()
            root.val += greater
            greater = root.val
            root = root.left

        return head


    # 235. Lowest Common Ancestor of a Binary Tree(Easy)
    # 二叉搜索树的最近公共祖先
    def lowestCommonAncester(self, root, p, q):
        while (root.val - p.val) * (root.val - q.val) > 0: # 同侧
            if root.val < p.val:
                root = root.right
            else:
                root = root.left

        return root


    # 530. Minimum Absolute Difference in BST(Easy)
    # 二叉树的最小绝对差
    def getMinmumDifference(self, root):
        if not root:
            return 0

        res = float('inf')
        prev = TreeNode(float('-inf'))
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if res > abs(prev.val - root.val):
                res = abs(prev.val - root.val)
            prev = root
            root = root.right

        return res





if __name__ == '__main__':
    s = Solution()

    root = TreeNode(0)
    l = TreeNode(1)
    r = TreeNode(2)
    root.left = l
    root.right = r

    res = s.diameterOfBinaryTree(root)
    print(res)


    # 99.
    node1 = TreeNode(1)
    node2 = TreeNode(2)
    node3 = TreeNode(3)
    node4 = TreeNode(4)
    node3.left = node1
    node3.right = node4
    node4.right = node2

    res = s.recoverTree(node3)


    # 404.
    node1 = TreeNode(3)
    node2 = TreeNode(9)
    node3 = TreeNode(20)
    node4 = TreeNode(15)
    node5 = TreeNode(7)
    node1.left = node2
    node1.right = node3
    node3.left = node4
    node3.right = node5

    res = s.sumOfLeftLeaves(node1)

    # 538.
    nodes = []
    for i in range(9):
        nodes.append(TreeNode(i))
    nodes[1].left = nodes[0]
    nodes[1].right = nodes[2]
    nodes[2].right = nodes[3]
    nodes[6].left = nodes[5]
    nodes[6].right = nodes[7]
    nodes[7].right = nodes[8]
    nodes[4].left = nodes[1]
    nodes[4].right = nodes[6]

    s.convertBST(nodes[4])
    print(res)