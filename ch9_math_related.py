import math
import random


class Solution(object):
    # 公倍数与公因数
    # 辗转相除法，求得2个数的最大公因数(greatest common divisor, gcd);
    # 将2个数相乘再除以最大公因数，即可得最小公倍数(least common multiple, lcm).
    def gcd(self, a, b):
        if b == 0:
            return a
        else:
            return self.gcd(b, a % b)

    def lcm(self, a, b):
        return a * b / self.gcd(a, b)

    # 通过拓展欧几里得算法(extended gcd)，同时求的a,b的最大公因数
    # 及他们的系数，ax + by = gcd(a, b)
    def xGCD(self, a, b, x, y):
        if not b:
            x = 1
            y = 0
            return a

        x1, y1, gcd = self.xGCD(b, a%b, x1, y1)
        x = y1
        y = x1 - (a/b)*y1
        return gcd


    # 质数
    # 质数又称素数，在大于1的自然数中，只能被1和ta本身整除
    # 注意：每一个数都可以分解成质数的乘积
    # 204. Count Primes(Easy)
    # 计算小于n的质数数量
    # 埃拉托斯特尼筛法，判断一个整数是否是质数
    # 并且它可以在判断一个整数n时，同时判断所有小于n的整数
    # 从 1 到 n 遍历，假设当前遍历到 m，则把所有小于 n 的、且是 m 的倍 数的整数标为和数;
    # 遍历完成后，没有被标为和数的数字即为质数。
    # 要得到自然数n以内的全部素数，必须把不大于 sqrt(n) 的所有素数的倍数剔除，剩下的就是素数。
    def countPrimes(self, n):
        if n <= 2:
            return 0

        prime = [True] * n
        i = 3
        sqrt_n = n ** 0.5
        count = n // 2
        while i <= sqrt_n:
            if prime[i]:
                for j in range(i*i, n, 2*i):
                    if prime[j]:
                        count -= 1
                        prime[j] = False
            i += 2

        return count


    # 数字处理
    # 504. Base 7(Easy)
    # 给定一个十进制整数，求它在七进制下的表示
    # 除法 + 取模
    def convertToBase7(self, num):
        if num == 0:
            return '0'

        is_negative = num < 0
        if is_negative:
            num = -1 * num
        ans = ''
        while num:
            a = num // 7
            b = num % 7
            ans = str(b) + ans
            num = a

        if is_negative:
            return '-' + ans
        else:
            return ans


    # 168. 26进制数字字符对应
    # 10进制转26进制，余数是0-25
    # 而本题的余数是1-26（对应a-z）
    # 那么每次都要消除这个差距
    def convertToTitle(self, n):
        ans = ''
        if n < 0:
            return ans

        while n:
            n -= 1
            a = n // 26
            b = n % 26
            ans = chr(b + 65) + ans
            n = a

        return ans


    # 172. Factorial Trailing Zeroes(Easy)
    # 给定一个非负整数，判断它的阶乘结果的结尾有几个0
    # method 1, 计算阶乘，统计0个数
    def trailingZeroes(self, n):
        if n < 2:
            return 0

        res = 1
        for i in range(2, n+1):
            res *= i

        res = list(str(res))
        count = 0
        for i in range(1, len(res)+1):
            if res[-1*i] != '0':
                break
            else:
                count += 1
        return count

    # 优化
    # 尾部的0由2x5得来，统计有多少个质因子5
    def trailingZeroes_primes(self, n):
        if n == 0:
            return 0
        else:
            return n // 5 + self.trailingZeroes_primes(n // 5)


    # 415. Add Strings(Easy)
    # 给定两个由数字组成的字符串，求它们相加的结果。
    def addStrings(self, num1, num2):
        m = len(num1)
        n = len(num2)
        if m > n:
            num2 = '0'*(m-n) + num2
        else:
            num1 = '0'*(n-m) + num1
        p = max(m, n) - 1
        ans = ''
        carry = 0

        while p >= 0:
            tmp = int(num1[p]) + int(num2[p]) + carry
            res = tmp % 10
            carry = tmp // 10
            ans = str(res) + ans
            p -= 1

        return str(int(str(carry) + ans))


    # 326. Power of Three(Easy)
    # 判断一个数字是否是 3 的次方
    # 1. if log_3^n = m, m是整数，那么n是3的次方（换底）
    # 2. int范围内，3的最大次方是3^19, 那么如果3^19 % n == 0, 那么n是3的次方
    def isPowerOfThree_log(self, n):
        # TODO: Debug
        if n <= 0:
            return False
        return (math.log(n) / math.log(3)) % 1 == 0

    def isPowerOfThree_div(self, n):
        return n > 0 and 3**19 % n == 0


    # 随机与取样
    # 384. Shuffle an Array(Medium)
    # fisher Yate 洗牌算法
    # class OpOnArray()

    # 528. Random Pick with Weight(Medium)
    # 给定一个数组，数组每个位置的值表示该位置的权重，要求按照权重的概率去随机采样。
    # TODO

    # 382. Linked List Random Node(Medium)
    # 给定一个单向链表，要求设计一个算法，可以随机取得其中的一个数字。


class OpOnArray(object):
    def __init__(self, nums):
        self.nums = nums

    def reset(self):
        return self.nums

    def shuffle(self):
        n = len(self.nums)
        if not n:
            return []
        shuffled = self.nums.copy()

        for i in range(n-1, -1, -1):
            pos = random.randrange(0, i+1)
            shuffled[i], shuffled[pos] = shuffled[pos], shuffled[i]

        return shuffled






if __name__ == '__main__':
    s = Solution()

    # gcd
    a, b = 1, 2
    res = s.gcd(a, b)

    # lcm
    res = s.lcm(a, b)

    # 质数
    # 204. Count Primes(Easy)
    n = 10000
    res = s.countPrimes(n)

    # 504. Base 7(Easy)
    n = 100
    res = s.convertToBase7(n)

    # 172. Factorial Trailing Zeroes(Easy)
    n = 5
    res = s.trailingZeroes(n)
    res = s.trailingZeroes_primes(n)

    # 415. Add Strings(Easy)
    num1 = '99'
    num2 = '0'
    res = s.addStrings(num1, num2)

    # 326. Power of Three(Easy)
    n = 27
    # res = s.isPowerOfThree_log(n)
    res = s.isPowerOfThree_div(n)

    # 384. Shuffle an Array(Medium)
    nums = [1, 2, 4]
    op_array = OpOnArray(nums)
    res = op_array.shuffle()

    # 168. 26进制转换字符串
    n = 27
    res = s.convertToTitle(n)

    print(res)