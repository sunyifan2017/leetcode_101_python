from collections import deque

class DP(object):
    # 基础动态规划：一维
    # 70. Climbing Stairs(Easy)
    # 爬楼梯, 经典的斐波那契数列
    # dp[i] = dp[i-1] + dp[i-2]
    def climbStairs(self, n):
        if n <= 2:
            return n

        dp = [1] * (n+1)
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

    # 上面用一维数组来记录状态
    # 因为dp[i]只与dp[i-1]和dp[i-2]相关
    # 可以用2个变量来实时更新存储dp[i-1]和dp[i-2]
    # 空间复杂度压缩： O(n) -> O(1)
    def climbStairs_space_optim(self, n):
        if n <= 2:
            return n

        dp1, dp2 = 2, 1
        for i in range(2, n):
            dp = dp1 + dp2
            dp2 = dp1
            dp1 = dp


    # 198. House Robber(Medium)
    # dp[i] = max(dp[i-1], nums[i] + dp[i-2])
    def rob(self, nums):
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)

        n = len(nums)
        dp = nums.copy()
        dp[1] = max(nums[:2])
        for i in range(2, n):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2])

        return max(dp)

    # 同样可以空间压缩
    # 代码pass


    # 413. Arithmetic Slices(Medium)
    # 等差数列：nums[i] - num[i-1] = nums[i-1] - nums[i-2]
    def numberOfArithmeticSlices(self, nums):
        if len(nums) < 2:
            return 0

        dp = [0] * len(nums)
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = 1 + dp[i-1]

        return sum(dp)


    # 基础动态规划：二维
    # 64. Minimum Path Sum(Medium)
    # dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])
        dp = [[0]*n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if i < 1 and j < 1:
                    dp[i][j] = grid[i][j]
                elif i < 1:
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                elif j < 1:
                    dp[i][j] = dp[i-1][j] + grid[i][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

        return dp[m-1][n-1]


    # 542. 01 Matrix(Medium)
    # BFS
    def updateMatrix(self, mat):
        m = len(mat)
        n = len(mat[0])
        visited = [[0]*n for _ in range(m)]
        queue = deque()

        # 取0节点
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append([i, j])
                    visited[i][j] = 1

        # bfs
        directions = [-1, 0, 1, 0, -1]
        while queue:
            n_nodes = len(queue)
            for i in range(n_nodes):
                [r, c] = queue.popleft()
                for k in range(4):
                    x = r + directions[k]
                    y = c + directions[k+1]
                    if x >= 0 and x < m and y >= 0 and y < n and not visited[x][y]:
                        mat[x][y] = 1 + mat[r][c]
                        queue.append([x, y])
                        visited[x][y] = 1

        return mat

    # DP
    # if mat[i][j] == 0: dp[i][j] = 0
    # if mat[i][j] == 1: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i+1][j], dp[i][j+1])
    def updateMatrix_DP(self, mat):
        m = len(mat)
        n = len(mat[0])
        dp = [[float('inf')]*n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    dp[i][j] = 0

        for i in range(m):
            for j in range(n):
                if i > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)

        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if i < m-1:
                    dp[i][j] = min(dp[i][j], dp[i+1][j] + 1)
                if j < n-1:
                    dp[i][j] = min(dp[i][j], dp[i][j+1] + 1)

        return dp


    # 221. Maximal Square(Medium)
    # dp[i][j] 以[i, j]为右下角的边长
    # if matrix[i][j] == 1: dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    # if matrix[i][j] == 0: dp[i][j] = 0
    def maximalSquare(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0]*n for _ in range(m)]
        max_side = 0

        for i in range(m):
            for j in range(n):
                if i < 1 and j < 1 and matrix[i][j] == '1':
                    dp[i][j] = 1
                elif i < 1 or j < 1:
                    if matrix[i][j] == '1':
                        dp[i][j] = 1
                else:
                    if matrix[i][j] != '0':
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

                max_side = max(max_side, dp[i][j])

        return max_side * max_side


    # 分割类型问题
    # 279. Perfect Squares(Medium)
    # dp[i]表示数字i最少可以由几个完全平方数相加构成
    # e.g. dp[15] = min(dp[14], dp[11], dp[6]) + 1
    # dp[i] = min(dp[i-1], dp[i-4], ..., dp[i-k^2]) + 1
    def numSquares(self, n):
        root = n ** 0.5
        if int(root) == root:
            return 1

        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        square_nums = [i**2 for i in range(1, int(root)+1)]
        for i in range(1, n+1):
            for square in square_nums:
                if square > i:
                    break
                dp[i] = min(dp[i], dp[i-square] + 1)

        return dp[n]


    # 91. Decode Ways(Medium)
    # dp[i]: s[:i]满足条件的解码方式总数
    # if [i-1]字符是0: dp[i] = dp[i-2]
    # elif [i]字符可以与前一个字符组合在一起
    #   (s[i] \in [1, 6] and s[i-1] == 2 or s[i] == 1):
    #       dp[i] = dp[i-1] + dp[i-2]
    # else: dp[i] = dp[i-1]
    # TODO: 复习
    def numDecodings(self, s):
        if s[0] == '0':
            return 0
        dp = [0] * (len(s) + 1)
        dp[0], dp[1] = 1, 1

        for i in range(1, len(s)):
            if s[i] == '0':
                if s[i - 1] == '1' or s[i - 1] == '2':
                    dp[i + 1] += dp[i - 1]
                else:
                    return 0
            elif s[i - 1] == '1' or (s[i - 1] == '2' and 1 <= int(s[i]) <= 6):
                dp[i + 1] = dp[i - 1] + dp[i]
            else:
                dp[i + 1] = dp[i]

        return dp[-1]


    # 139. Word Break
    # dp[i]: 截止到字符串s的第i位，能否分割
    # dp[i] = dp[i-len(wordList[1])] or dp[i-len(wordList[2])] or ...
    def wordBreak(self, s, wordDict):
        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True

        for i in range(1, n+1):
            for word in wordDict:
                if i >= len(word) and s[i-len(word): i] == word:
                    dp[i] = dp[i] or dp[i-len(word)]

        return dp[-1]


    # 子序列问题
    # 按照 LeetCode 的习惯，子序列(subsequence)不必连续，
    # 子数组(subarray)或子字符串 (substring)必须连续。

    # 300. Longest Increasing Subsequence(Medium)
    # TODO: 二分查找
    def lengthOfLIS(self, nums):
        n = len(nums)
        max_length = 0
        if n < 2:
            return n

        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], 1 + dp[j])

            max_length = max(max_length, dp[i])

        return max_length


    # 1143. Longest Common Subsequence(Medium)
    # dp[i][j]: text1[: i]和text2[: j]的最长公共子序列
    # if text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    # else: dp[i] = max(dp[i-1][j], dp[i][j-1])
    def longestCommonSubsquence(self, text1, text2):
        m = len(text1)
        n = len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[-1][-1]


    # 背包问题
    # item具有2个特征，平衡2个特征，得到最优解
    # dp为二维数组
    # 01背包，剩余空间时，不可选当前物品，dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]]+v[i])
    # 01背包，压缩空间，逆序
    # 无限背包，剩余空间时，可选当前物品，dp[i][j] = max(dp[i-1][j], dp[i][j-w[i]]+v[i])
    # 无限背包，压缩空间，正序

    # 416. Partition Equal Subset Sum(Medium)
    # 同01背包，背包大小为 sum // 2
    # 没有价值，只有重量
    # 能不能放下这个物品
    # 如果能放下，能不能刚好 == 背包容量
    def canPartition(self, nums):
        target = sum(nums) // 2
        mod = sum(nums) % 2
        n = len(nums)
        if 1 == mod:
            return False

        dp = [[False]*(target+1) for _ in range(n+1)]
        dp[0][0] = True

        for i in range(1, n+1):
            for j in range(1, target+1):
                if nums[i-1] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]

        return dp[-1][-1]

    # 空间压缩
    def canPartition_space_optim(self, nums):
        target = sum(nums) // 2
        mod = sum(nums) % 2
        if 1 == mod:
            return False

        dp = [False for _ in range(target+1)]
        dp[0] = True

        for i in range(len(nums)):
            for j in range(target, 0, -1):
                if nums[i-1] > j:
                    pass
                else:
                    dp[j] = dp[j] or dp[j-nums[i-1]]

        return dp[-1]


    # 474. Ones and Zeroes(Medium)
    # 能不能放，能放的话，维持最大子集，并符合条件
    # 2个背包，一个m，一个n
    def findMaxForm(self, strs, m, n):
        dp = [[[0]*(n+1) for _ in range(m+1)] for _ in range(len(strs)+1)]

        for i in range(1, len(strs)+1):
            amount_0, amount_1 = self.count_0_1(strs[i-1])
            for j in range(m+1):
                for k in range(n+1):
                    if amount_0 > j or amount_1 > k:
                        dp[i][j][k] = dp[i-1][j][k]
                    else:
                        dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][j-amount_0][k-amount_1]+1)
        return dp[-1][-1][-1]

    def count_0_1(self, strs):
        amount_0 = 0
        amount_1 = 0
        for s in strs:
            if s == '0':
                amount_0 += 1
            else:
                amount_1 += 1
        return amount_0, amount_1

    # 空间压缩
    def findMaxForm_space_optim(self, strs, m, n):
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(len(strs)):
            amount_0, amount_1 = self.count_0_1(strs[i])
            for j in range(m, amount_0-1, -1):
                for k in range(n, amount_1-1, -1):
                    dp[j][k] = max(dp[j][k], dp[j-amount_0][k-amount_1]+1)

        return dp[-1][-1]


    # 322. Coin Change(Medium)
    # 无限背包
    # 正序，循环颠倒
    def coinChange(self, coins, amount):
        if not coins:
            return -1

        dp = [amount + 1] * (amount + 1)    # 取最小量，初始化要大一些
        dp[0] = 0                           # 注意初始条件

        for i in range(1, amount+1):
            for coin in coins:
                if coin > i:
                    pass
                else:
                    dp[i] = min(dp[i], dp[i-coin]+1)

        if dp[-1] == amount+1:
            return -1
        else:
            return dp[-1]


    # 字符串编辑
    # 72. Edit Distance(Hard)
    # word1 -> word2
    # dp[i][j]: word1[:i]修改为word2[:j]需要的操作数
    # if word1[i]不需要修改: dp[i][j] = dp[i-1][j-1]
    # else: min(1. dp[i][j] = dp[i-1][j-1] + 1 改
    #           2. dp[i][j] = dp[i-1][j] + 1   增加
    #           3. dp[i][j] = dp[i][j-1] + 1   删
    def minDistance(self, word1, word2):
        n = len(word1)
        m = len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif word1[j-1] == word2[i-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

        return dp[-1][-1]


    # 650. 2 Keys Keyboard(Medium)
    # dp[i]: 延展到i需要的最少操作数
    # i    : 2
    # dp[i]: 2 cp
    # i    : 3
    # dp[i]: 3 cpp
    # ...
    # i    : 1 2 3 4 5 6 7 8 9 10
    # dp[i]: 1 2 3 4 5 5 7 6 6 7
    # if k是质数：dp[k] = k
    # else: k = a * b
    # dp[k] = dp[a] + dp[b]
    def minStep(self, n):
        dp = [0] * (n+1)
        for i in range(2, n+1):
            dp[i] = i
            for j in range(2, i):
                if i % j == 0:
                    dp[i] = dp[j] + dp[i // j]
                    break

        return dp[-1]


    # 10.Regular Expression Matching(Hard)
    def isMatch(self, s, p):
        n = len(p)
        m = len(s)
        dp = [[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True

        for i in range(1, n+1):
            if p[i-1] == '*':
                dp[0][i] = dp[0][i-2]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] != '*':
                    dp[i][j] = dp[i-1][j-1] and p[j-1] == s[i-1]
                elif p[j-2] != '.' and p[j-2] != s[i-1]:  # 删
                    dp[i][j] = dp[i][j-2]
                else:
                    dp[i][j] = dp[i][j-1] or dp[i-1][j] or dp[i][j-2]   # pass or 增 or 删

        return dp[-1][-1]


    # 股票交易
    # 状态机

    # 121. Best Time to Buy ans Sell Stock(Easy)
    # 只买卖一次，收益最大
    # 状态：没有股票0，持有股票1
    # 初始：0，结束：0
    #        0 - prices[i]
    #        ---------->
    # 0 <-- 0             1 --> 1
    #        <----------
    #        p + prices[i]
    # 转一次
    # dp.size = [len(prices), 2] -> 0状态，1状态
    # dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
    # dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]), dp[i-1][0] = 0
    def maxProfit(self, prices):
        n = len(prices)
        dp = [[0]*2 for _ in range(n+1)]
        dp[0][0] = 0
        dp[0][1] = -float('inf')

        for i in range(1, n + 1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i - 1])
            dp[i][1] = max(dp[i-1][1], 0 - prices[i - 1])

        return dp[-1][0]

    def maxProfit_normal(self, prices):
        sell = 0
        buy = float('inf')
        for i in range(len(prices)):
            buy = min(buy, prices[i])
            sell = max(sell, prices[i]-buy)
        return sell


    # 122. Best Time to Buy and Sell Stock II(Easy)
    # 买卖无限次，收益最大
    # 状态：没有股票0，持有股票1
    # 初始：0，结束：0
    #        p - prices[i]
    #        ---------->
    # 0 <-- 0             1 --> 1
    #        <----------
    #        p + prices[i]
    # 转无数次
    # dp.size = [len(prices), 2] -> 0状态，1状态
    # dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
    # dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    def maxProfit_II(self, prices):
        n = len(prices)
        dp = [[0]*2 for _ in range(n+1)]
        dp[0][0] = 0
        dp[0][1] = -float('inf')

        for i in range(1, n+1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i-1])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i-1])

        return dp[-1][0]


    # 188. Best Time to Buy and Sell Stock IV(Hard)
    # 买卖k次，收益最大
    # 状态：没有股票0，持有股票1
    # 初始：0，结束：0
    #        0 - prices[i]
    #        ---------->
    # 0 <-- 0             1 --> 1
    #        <----------
    #        p + prices[i]
    # 转k次
    # dp.size = [len(prices), 2, k] -> 0状态，1状态
    # dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
    # dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    def maxProfit_IV(self, prices, k):
        n = len(prices)
        dp = [[[0]*2 for _ in range(n+1)] for _ in range(k)]

        if not dp:
            return 0

        for i in range(k):
            dp[i][0][0] = 0
            dp[i][0][1] = -float('inf')

        for j in range(1, n+1):
            dp[0][j][0] = max(dp[0][j-1][0], dp[0][j-1][1] + prices[j-1])
            dp[0][j][1] = max(dp[0][j-1][1], 0 - prices[j-1])

        for i in range(1, k):
            for j in range(1, n+1):
                dp[i][j][0] = max(dp[i][j-1][0], dp[i][j-1][1] + prices[j-1])
                dp[i][j][1] = max(dp[i][j-1][1], dp[i-1][j-1][0] - prices[j-1])

        return dp[-1][-1][0]


    # 309. Best Time to Buy and Sell Stock with Cooldown(Medium)
    # 买卖无限次，收益最大
    # 状态：没有股票0，持有股票1，冷却期2
    # 初始：0，结束：0 or 2
    #                   P + prices[i]
    #        <----- 2 <-----
    #       |               |
    # 0 <-- 0               1 --> 1
    #         ------------->
    #         p - prices[i]
    # 转无数次
    # dp.size = [len(prices), 3] -> 0状态，1状态, 2状态
    # dp[i][0] = max(dp[i-1][0], dp[i-1][2])
    # dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
    # dp[i][2] = dp[i-1][1] + prices[i]
    def maxProfit_cooldown(self, prices):
        n = len(prices)
        dp = [[0]*3 for _ in range(n+1)]
        dp[0][0] = 0
        dp[0][1] = -float('inf')
        dp[0][2] = 0

        for i in range(1, n+1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][2])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i-1])
            dp[i][2] = dp[i-1][1] + prices[i-1]

        return max(dp[-1][0], dp[-1][2])


    # Basic Excises
    # 213. House Robber II(Medium)
    # 环形数组
    # 分情况讨论，1. 1号房偷，则最后的房不能偷
    # 2. 1号房不偷，则最后的房可偷，可不偷
    def rob_II(self, nums):
        n = len(nums)
        if n < 3:
            return max(nums)

        # Choose No.1
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = nums[0]
        for i in range(2, n-1):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        res = max(dp)

        # Not choose No.1
        dp = [0] * n
        dp[0] = 0
        dp[1] = nums[1]
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])

        res = max(res, max(dp))

        return res


    # 53. Maximum Subarray(Easy)
    # dp[i] = max(dp[i-1]+nums[i], nums[i])
    def maxSubarray(self, nums):
        n = len(nums)

        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1]+nums[i], nums[i])

        return max(dp)

    # 343. Integer Break(Medium)
    # dp[i] = max({1*dp[i-1], 1*(i-1)}, {2*dp[i-2], 2*(i-2)}, 3*dp[i-3], ..., {(i-1)*dp[1], (i-1)*1})
    def integerBreak(self, n):
        if n < 3:
            return 1

        dp = [0] * (n+1)

        for i in range(2, n+1):
            for j in range(1, i):
                dp[i] = max(dp[i], j * dp[i-j], j * (i-j))

        return dp[-1]

    # 优先拆成3的倍数
    # 其次拆成2的倍数
    # 最次拆成1
    def interBreak_math(self, n):
        if n <= 3:
            return n-1

        exponent, remainder = n // 3, n % 3
        if 0 == remainder:
            return 3 ** exponent
        elif 1 == remainder:
            return 3 ** (exponent - 1) * 4
        elif 2 == remainder:
            return 3 ** exponent * 2


    # 583. Delete Operation for Two Strings(Medium)
    # 最长公共子序列
    # if word1[i] == word2[j]: dp[i][j] = dp[i-1][j-1] + 1
    # else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    # res = len(word1) - dp[-1][-1] + len(word2) - dp[-1][-1]
    # * OR 从操作数的角度
    def minDistance_only_delete(self, word1, word2):
        m = len(word1)
        n = len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return m + n - 2*dp[-1][-1]


    # Harder Excises
    # 646. Maximum Length of Pair Chain(Medium)
    # 最长递增子序列的变种
    # 对于nums[i], 对于i之前的j，如果nums[i] > nums[j]: dp[i] = 1 + dp[j], 2层遍历
    def findLongestChain(self, pairs):
        n = len(pairs)
        if n < 2:
            return n
        pairs = sorted(pairs, key= lambda x: x[1])
        dp = [0] * n
        dp[0] = 1

        for i in range(n):
            for j in range(i):
                if pairs[i][0] > pairs[j][1]:
                    dp[i] = max(dp[i], 1 + dp[j])

        return max(dp)

    # 贪心
    def findLongestChain_greedy(self, pairs):
        n = len(pairs)
        if n < 2:
            return n

        pairs = sorted(pairs, key=lambda x: x[1])
        prev = pairs[0]
        res = 1

        for i in range(1, n):
            if pairs[i][0] > prev[1]:
                res += 1
                prev = pairs[i]

        return res


    # 376. Wiggle Subsquence(Medium)
    # 摆动序列，[1, 7, 4, 9, 2, 5] -> [6, -3, 5, -7, 3]
    # nums[i-1] > nums[i], nums[i+1] > nums[i]
    # OR nums[i-1] < nums[i], nums[i+1] < nums[i]
    # dp[i][s]：表示以nums[i]为结尾的符合条件的最长子序列长度
    # s表示状态，1表示上升，0表示下降
    def wiggleMaxLength(self, nums):
        n = len(nums)
        dp = [[0]*2 for _ in range(n)]
        dp[0][0], dp[0][1] = 1, 1
        max_length = 1

        for i in range(1, n):
            for j in range(i):
                if nums[i] != nums[j]:
                    s = nums[i] > nums[j]
                    dp[i][s] = max(dp[i][s], dp[j][not s]+1)
            max_length = max(max_length, dp[i][0], dp[i][1])

        return max_length


    # 贪心？归纳
    # 波峰，波谷保留，一定是震荡的
    # 梯度正负不变的，节点合并
    #              O
    #           o     o    O
    #    O   o          O     O
    # O    O
    # 1 17 5 10 13 15 10 5 16 8
    def wiggleMaxlength_greedy(self, nums):
        res = 1
        n = len(nums)
        if n == 1:
            return res

        pre_grad = 0
        for i in range(1, n):
            grad = nums[i] - nums[i-1]
            if pre_grad == 0:
                pre_grad = -grad

            if pre_grad * grad < 0:
                res += 1
                pre_grad = grad

        return res


    # 494. Target Sum(Medium)
    # dp[i][j] 表示以nums[i]为结尾的和为j的表达数量
    # dp[i][j] = dp[i-1][j-nums[i]] + dp[i-1][j+nums[j]]
    def findTargetSumWays(self, nums, target):
        pass


    # 714. Best Time to Buy and Sell with Transaction Fee(Medium)
    # 初始0
    # 结束0
    def maxProfit_with_fee(self, prices, fee):
        n = len(prices)
        if n < 2:
            return 0

        dp = [[0]*2 for _ in range(n+1)]
        dp[0][0] = 0
        dp[0][1] = -float('inf')

        for i in range(1, n+1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i-1])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i-1]-fee)

        return dp[-1][0]

if __name__ == '__main__':
    s = DP()

    # 一维
    # 70. climbStairs(Easy)
    # res = s.climbStairs(3)

    # 198. House Robber(Medium)
    # nums = [2,1,1,2]
    # res = s.rob(nums)

    # 413. Arithmetic Slices(Medium)
    # nums = [1, 2, 3, 4]
    # res = s.numberOfArithmeticSlices(nums)

    # 二维
    # 64. Minimum Path Sum(Medium)
    # grid = [[1,3,1],
    #         [1,5,1],
    #         [4,2,1]]
    # res = s.minPathSum(grid)

    # 542. 01 Matrix(Medium)
    # matrix = [[0,0,0],
    #          [0,1,0],
    #          [0,0,0]]
    # matrix = [[0,0,0],
    #         [0,1,0],
    #         [1,1,1]]
    # matrix = [[0],[0],[0],[0],[0]]
    # matrix = [[0,1,0,1,1],[1,1,0,0,1],[0,0,0,1,0],[1,0,1,1,1],[1,0,0,0,1]]
    # res = s.updateMatrix(matrix)
    # res = s.updateMatrix_DP(matrix)

    # 221. Maximal Square(Medium)
    # matrix = [["1","0","1","0","0"],
    #         ["1","0","1","1","1"],
    #         ["1","1","1","1","1"],
    #         ["1","0","0","1","0"]]
    # res = s.maximalSquare(matrix)

    # 分割类型问题
    # 279. Perfect Squares(Medium)
    # n = 1
    # res = s.numSquares(n)

    # 91. Decode Ways(Medium)
    # string = '110'
    # res = s.numDecodings(string)

    # 139. Word Break(Medium)
    # string = "applepenapple"
    # wordList = ["apple", "pen"]
    # string = "catsandog"
    # wordList = ["cats","dog","sand","and","cat"]
    # res = s.wordBreak(string, wordList)

    # 300. Longest Increasing Subsequence(Medium)
    # nums = [10,9,2,5,3,7,101,18]
    # res = s.lengthOfLIS(nums)


    # 背包问题
    # 416. Partition Equal Subset Sum(Medium)
    # nums = [1, 5, 11, 5]
    # nums = [1, 2, 3, 5]
    # res = s.canPartition(nums)

    # 474. Ones and Zeroes(Medium)
    # strs = ["10", "0001", "111001", "1", "0"]
    # m = 5
    # n = 3
    # strs = ["10","0","1"]
    # m = 1
    # n = 1
    # res = s.findMaxForm(strs, m, n)
    # res = s.findMaxForm_space_optim(strs, m, n)

    # 322. Coins Change(Mediun)
    # coins = [1, 2, 5]
    # amount = 11
    # coins = [1]
    # amount = 2
    # res = s.coinChange(coins, amount)


    # 字符串编辑
    # 72. Edit Distance(Hard)
    # word1 = 'horse'
    # # word2 = 'ros'
    # word1 = "intention"
    # word2 = "execution"
    # res = s.minDistance(word1, word2)

    # 650. 2 Keys Keyboard(Medium)
    # n = 10
    # res = s.minStep(n)

    # 10.Regular Expression Matching(Hard)
    # strs = 'missi'
    # p = 'mi.*'
    # strs = 'aa'
    # p = 'a*'
    # strs = 'ab'
    # p = '.*'
    # strs = "mississippi"
    # p = "mis*is*p*."
    # res = s.isMatch(strs, p)

    # 121. Best Time to Buy ans Sell Stock(Easy)
    # 122. Best Time to Buy and Sell Stock II(Easy)
    # prices = [7, 1, 5, 3, 6]
    # prices = [3, 2, 6, 5, 0, 3]
    # prices = [1, 2, 3, 0, 2]
    # k = 0
    # res = s.maxProfit(prices)
    # res = s.maxProfit_normal(prices)
    # res = s.maxProfit_II(prices)
    # res = s.maxProfit_IV(prices, k)
    # res = s.maxProfit_cooldown(prices)


    # Basic Excises
    # 213. House Robber II(Medium)
    # nums = [1, 3, 1, 3, 100]
    # nums = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    # res = s.rob_II(nums)

    # 53. Maximum Subarray(Easy)
    # nums = [-2,1,-3,4,-1,2,1,-5,4]
    # nums = [-1]
    # res = s.maxSubarray(nums)

    # 343. Integer Break(Medium)
    # n = 6
    # res = s.integerBreak(n)

    # 583. Delete Operation for Two Strings(Medium)
    # word1 = 'sea'
    # word2 = 'eat'
    # res = s.minDistance_only_delete(word1, word2)

    # Harder Excises
    # 646. Maximum Length of Pair Chain(Medium)
    # pairs = [[1,2], [2,3], [3,4]]
    # pairs = []
    # pairs = [[1,2],[7,8],[4,5]]
    # pairs = [[-6,9],[1,6],[8,10],[-1,4],[-6,-2],[-9,8],[-5,3],[0,3]]
    # res = s.findLongestChain(pairs)
    # res = s.findLongestChain_greedy(pairs)

    # 376. Wiggle Subsquence(Medium)
    # nums = [1, 7, 4, 9, 2, 5]
    # nums = [1, 1, 17, 17, 5, 10, 13, 15, 10, 5, 16, 8]
    # res = s.wiggleMaxLength(nums)
    # res = s.wiggleMaxlength_greedy(nums)

    # 494. Target Sum(Medium)
    # nums = [1, 1, 1, 1, 1]
    # target = 3
    # res = s.findTargetSumWays(nums, target)
    # print(res)