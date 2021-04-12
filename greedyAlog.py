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

        # from left to right
        for i in range(n-1):
            if ratings[i] > ratings[i+1]:
                res[i] += 1
        
        for i in range(n-1, 0, -1):
            if ratings[i] > ratings[i-1]:
                res[i] += 1

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


    # Basic Excise
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


    # Basic Excise
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

    
    # Basic Excise
    # 763. Partition Labels(Medium)
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

        acount = 1
        prev = labels[0]
        res = [prev]

        for i in range(1, len(labels)):
            if labels[i][0] >= prev[1]:
                acount += 1
                res.append(prev)
                prev = labels[i]
            else:
                prev[-1] = max(prev[-1], labels[i][1])
                res[-1] = prev
        
        return [i[1] - i[0] + 1 for i in res]


    # Basic Excise
    # 122. Best Time to Buy and Sell Stock II(Easy)
    """
    Emmmm, pn - p1 = (p2 - p1) + (p3 - p2) + ... + (pn - pn-1)
    """
    def maxProfit(self, prices):
        n = len(prices)
        profit = 0
        for i in range(1, n):
            if prices[i] > prices[i-1]:
                profit += (prices[i] - prices[i-1])
        
        return profit

    
    # Harder Excise
    # 406. Queue Reconstruction by Height(Medium)
    """假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。
    每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

    请你重新构造并返回输入数组 people 所表示的队列。
    返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

     

    示例 1：

    输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    解释：
    编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
    编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
    编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
    编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
    编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
    示例 2：

    输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
    输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]"""
    def reconstructQueue(self, people):
        pass

    

if __name__ == '__main__':
    s = Solution()
    # res = s.findContentChildren([2, 1], [1, 2, 3])
    # res = s.candy([1, 0, 2])
    # res = s.areasOverlapIntervals([[1, 2], [2, 4], [1, 3]])
    # res = s.canPlaceFlowers([1, 0, 0, 0, 1], 2)
    # res = s.findMinArrowShots([[2,3],[2,3]])
    # res = s.partitionLabels("ababcbacadefegdehijhklij")
    # res = s.maxProfit([1, 2, 3, 4, 5])
    res = s.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])

    print(res)