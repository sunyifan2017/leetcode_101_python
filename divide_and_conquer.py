class Solution(object):
    # 表达式问题
    # 241. Different Ways to Add Parenthese(Medium)
    # 为运算表达式设计优先级
    def diffWaysToCompute(self, expression):
        n = len(expression)
        ways = []
        for i in range(n):
            c = expression[i]
            if c == '+' or c == '-' or c == '*':
                left = self.diffWaysToCompute(expression[: i])
                right = self.diffWaysToCompute(expression[i+1:])

                for l in left:
                    for r in right:
                        if c == '+':
                            ways.append(l+r)
                        elif c == '-':
                            ways.append(l-r)
                        elif c == '*':
                            ways.append(l*r)

        if not ways:
            ways.append(int(expression))
        return ways


    # Basic Excises
    # 932. Beautiful Array(Medium)
    def beautifulArray(self, n):
        memo = {1: [1]}
        def f(N):
            if N not in memo:
                odds = f((N+1)/2)
                evens = f(N/2)
                memo[N] = [2*x-1 for x in odds] + [2*x for x in evens]
            return memo[N]
        return f(n)



if __name__ == '__main__':
    s = Solution()
    expression = "2-1-1"
    res = s.diffWaysToCompute(expression)
    print(res)
