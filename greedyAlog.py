class Solution(object):
    # 455. Assign Cookies(Easy) 
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


    # 435. Non-overlapping Intervals(Medium)
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
    def canPlaceFlowers(self, flowerbed, n):
        flowerbed = [0] +  flowerbed + [0]
        avaliable = 0
        for i in range(1, len(flowerbed)-1):
            if flowerbed[i-1] == 0 and flowerbed[i+1] == 0 and flowerbed[i] != 1:
                flowerbed[i] = 1
                avaliable += 1

        if avaliable >= n:
            return True
        else:
            return False


    # Basic Excises
    # 452. Minimum Number of Arrows to Burst Balloons(Medium)
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
    # 665. Non-decreasing Array(Easy)
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
    # res = s.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])
    res = s.checkPossibility([4, 2, 3])

    print(res)