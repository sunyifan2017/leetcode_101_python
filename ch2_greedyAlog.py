"""
贪心算法采用贪心的策略，保证每次操作都是局部最优
且局部结果互不干涉/独立
从而使最后得到的结果

贪心经常会涉及到排序
"""
class Solution(object):
    # 分配问题
    # 455. Assign Cookies(Easy)
    # 分发饼干
    # 让吃的少的小孩先吃饱
    def findContentChildren(self, children, cookies):
        children.sort()
        cookies.sort()

        child = 0
        cookie = 0

        while child < len(children) and cookie < len(cookies):
            if children[child] <=  cookies[cookie]:
                child += 1
                cookie += 1
            else:
                cookie += 1

        return child


    # 135. Candy(hard)
    # 分发糖果
    # 每个孩子至少分配到1个糖果
    # 评分高的孩子比他两侧的孩子获得更多的糖果
    def candy(self, ratings):
        n = len(ratings)
        if n < 2:
            return n

        res = [1] * n

        # 两次遍历，每次都要跟不再变换的值做比较
        # from left to right
        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                res[i] = res[i-1] + 1

        # from right to left
        for i in range(n-1, 0, -1):
            if ratings[i-1] > ratings[i]:
                res[i-1] = max(res[i-1], res[i]+1)

        return sum(res)


    # 区间问题
    # 435. Non-overlapping Intervals(Medium)
    # 无重叠区间，找到需要移除区间的最小数量，使剩余区间不重复
    # 等价于尽量多不重叠的区间
    # 选择的区间结尾越小，余留给其他区间的空间就越大，能保留更多的区间
    def areasOverlapIntervals(self, intervals):
        if len(intervals) <= 1:
            return 0

        intervals = sorted(intervals, key=lambda x: x[1])

        n = len(intervals)
        removed = 0
        prev = intervals[0]

        for i in range(1, n):
            if intervals[i][0] >= prev[1]:
                prev = intervals[i]
            else:
                removed += 1

        return removed


    # Basic Excises
    # 605. Can Place Flowers(Easy)
    # 种花问题，话不能种在相邻位置
    # 能否种n朵花
    def canPlaceFlowers(self, flowerbed, n):
        flowerbed = [0] +  flowerbed + [0]
        avaliable = 0
        for i in range(1, len(flowerbed)-1):
            if flowerbed[i-1] == 0 and flowerbed[i+1] == 0 and flowerbed[i] == 0:
                flowerbed[i] = 1
                avaliable += 1

        if avaliable >= n:
            return True
        else:
            return False


    # Basic Excises
    # 452. Minimum Number of Arrows to Burst Balloons(Medium)
    # 用最少数量的箭引爆气球
    # 同区间问题
    # 尽量保留重叠区间
    # 所有气球中右边界位置最靠左的那一个，那么一定有一支箭的射出位置就是它的右边界
    # 否则就没有箭可以将其引爆了
    # 将这支箭引爆的所有气球移除
    def findMinArrowShots(self, points):
        n = len(points)
        if n <= 0:
            return n

        points = sorted(points, key=lambda x: x[1])
        arrows = 1
        prev = points[0]

        for i in range(1, n):
            if prev[1] >= points[i][0]:
                pass
            else:
                arrows += 1
                prev = points[i]

        return arrows


    # Basic Excises
    # 763. Partition Labels(Medium)
    # 划分字母区间
    # 为了满足贪心策略，需要预处理
    def partitionLabels(self, S):
        n = len(S)
        points = {}

        for i in range(n):
            if S[i] in points:
                points[S[i]][-1] = i
            else:
                points[S[i]] = [i, i]
        labels = list(points.values())
        labels = sorted(labels, key=lambda x: x[0])

        res = [labels[0]]

        for i in range(1, len(labels)):
            if labels[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], labels[i][1])
            else:
                res.append(labels[i])

        return [i[1] - i[0] + 1 for i in res]


    # Basic Excises
    # 122. Best Time to Buy and Sell Stock II(Easy)
    """
    分情况：
    1. 单独交易日：今天买入，明天卖出，收益：p2 - p1;
    2. 连续上涨日：今天买入，最后一天卖出，收益：pn - p1 = (p2 - p1) + (p3 - p2) + ... + (pn - pn-1)
                等价于每天都在买卖；
    3. 连续下降日：不买.

    可见，每天都要计算收益.
    """
    def maxProfit(self, prices):
        n = len(prices)
        profit = 0

        for i in range(1, n):
            if prices[i] > prices[i-1]:
                profit += (prices[i] - prices[i-1])

        return profit


    # Harder Excises
    # 406. Queue Reconstruction by Height(Medium)
    # 根据身高重建队列
    def reconstructQueue(self, people):
        people = sorted(people, key=lambda x: (x[0], -x[1])) # cow beer
        n = len(people)
        ans = [[] for i in range(n)]

        for person in people:
            # 第i个人的位置，必须在队列从左往右数第k_{i+1}个空位置
            spaces = person[1] + 1
            for i in range(n):
                if not ans[i]:
                    spaces -= 1
                    if spaces == 0:
                        ans[i] = person
                        break

        return ans


    # Harder Excises
    # 665. Non-decreasing Array(Medium)
    # 非递减数列
    # 最多改变一个元素，能否将数组变为非递减数组
    def checkPossibility(self, nums):
        if len(nums) == 1:
            return True

        nums = [-10 ** 6] + nums #+ [10 ** 6]
        modify_num = 0

        for i in range(2, len(nums)):
            if nums[i] < nums[i - 1]:
                # nums[i - 1] = nums[i]
                modify_num += 1
                if nums[i] >= nums[i - 2]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]

            if modify_num > 1:
                return False

        return True


if __name__ == '__main__':
    s = Solution()
    # res = s.findContentChildren([2, 1], [1, 2, 3])
    # res = s.candy([1,2,3,4,5])
    # res = s.areasOverlapIntervals([[1,2],[2,3],[3,4],[1,3]])
    # res = s.canPlaceFlowers([1, 0, 0, 0, 1], 2)
    # res = s.findMinArrowShots([[2,3],[2,3]])
    # res = s.partitionLabels("eaaaabaaec")
    # res = s.maxProfit([1, 2, 3, 4, 5])
    res = s.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])
    # res = s.checkPossibility([4, 2, 3])

    print(res)