import copy

class DFS(object):
    # 695. Max Area of Island(Medium)
    # stack style
    # no visited list
    def maxAreaOfIsland_stack(self, grid):
        # m行n列
        m = len(grid)
        n = len(grid[0])
        area = 0
        island = []
        direction = [-1, 0, 1, 0, -1]

        # 遍历搜索所有位置，判断是否开始搜索
        # 共mxn个节点
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    local_area = 1
                    grid[i][j] = 0
                    island.append([i, j])
                    while island:
                        [r, c] = island.pop()
                        for k in range(4):
                            x = r + direction[k]
                            y = c + direction[k+1]
                            if x >= 0 and x < m and y >=0 and y < n and grid[x][y] == 1:
                                grid[x][y] = 0
                                local_area += 1
                                island.append([x, y])

                    area = max(area, local_area)

        return area

    # stack style
    # with visited list
    def maxAreaOfIsland_stack_with_visited_list(self, grid):
        m = len(grid)
        n = len(grid[0])
        direction = [-1, 0, 1, 0, -1]
        visited = [[0]*n for _ in range(m)]
        area = 0
        island = []

        for i in range(m):
            for j in range(n):
                if grid[i][j] and not visited[i][j]:
                    local_area = 1
                    island.append([i, j])
                    visited[i][j] = 1
                    while island:
                        [r, c] = island.pop()
                        for k in range(4):
                            x = r + direction[k]
                            y = c + direction[k+1]
                            if x >= 0 and x < m and y >= 0 and y < n and grid[x][y] and not visited[x][y]:
                                local_area += 1
                                island.append([x, y])
                                visited[x][y] = 1
                    area = max(local_area, area)

        return area

    # recursion style
    # 主函数遍历所有搜索位置，判断是否搜索
    # 辅函数负责搜索
    def maxAreaOfIsland_recuresion_1(self, grid):
        if (not grid) or (not grid[0]):
            return 0
        
        m = len(grid)
        n = len(grid[0])
        max_area = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    max_area = max(max_area, self.dfs_island(grid, i, j))
        
        return max_area

    def dfs_island(self, grid, r, c, method=2):
        # 辅函数 method 1: 先判定是否越界，再进步一搜索
        if method == 1:
            if grid[r][c] == 0:
                return 0
            
            m = len(grid)
            n = len(grid[0])
            area = 1
            grid[r][c] = 0
            direction = [-1, 0, 1, 0, -1]
            for k in range(4):
                x = r + direction[k]
                y = c + direction[k+1]
                if x >= 0 and x < m and y >= 0 and y < n:
                    area += self.dfs_island(grid, x, y)
            return area

        # 辅函数 method 2: 先下一步搜索，待下一步搜索时，再判断合法性
        if method == 2:
            m = len(grid)
            n = len(grid[0])
            if r < 0 or r >= m or c < 0 or c>=n or grid[r][c] != 1:
                return 0

            grid[r][c] = 0
            return 1 + self.dfs_island(grid, r+1, c, 2) + \
                self.dfs_island(grid, r, c+1, 2) + \
                self.dfs_island(grid, r-1, c, 2) + \
                self.dfs_island(grid, r, c-1, 2)
            

    # 547. Friend Circles(Medium)
    # 输入是nxn的二维数组
    def findCircleNum(self, friends):
        n = len(friends)
        count = 0
        visited = [0 for _ in range(n)] # i和j是朋友，j和i是朋友

        # 遍历搜索所有位置，判断是否开始搜索
        # 共n个节点
        for i in range(n):
            if not visited[i]:
                # 找到隶属于统一朋友圈的所有节点，并将这些节点进行标记，无返回值
                self.dfs_friend(friends, i, visited)
                count += 1

        return count

    def dfs_friend(self, friends, i, visited):
        visited[i] = 1
        n = len(friends)
        for k in range(n):
            if friends[i][k] == 1 and not visited[k]:
                self.dfs_friend(friends, k, visited)


    def findCircleNum_stack(self, friends):
        n = len(friends)
        count = 0
        visited = [0] * n
        stack = []

        for i in range(n):
            if not visited[i]:
                visited[i] = 1
                stack.append(i)
                while stack:
                    person = stack.pop()
                    for k in range(n):
                        if friends[person][k] and not visited[k]:
                            stack.append(k)
                            visited[k] = 1
                count += 1

        return count


    # 417. Pacific Atlantic Water Flow (Medium)
    # 递归，外循环遍历边界节点，递归遍历上下左右邻居
    def pacificAtlantic(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        if not m or not n:
            return []

        ans = []
        can_reach_p = [[0]*n for _ in range(m)]
        can_reach_a = [[0]*n for _ in range(m)]
        
        for i in range(m):
            self.dfs_sea(matrix, can_reach_p, i, 0)
            self.dfs_sea(matrix, can_reach_a, i, n-1)

        for i in range(n):
            self.dfs_sea(matrix, can_reach_p, 0, i)
            self.dfs_sea(matrix, can_reach_a, m-1, i)

        for i in range(m):
            for j in range(n):
                if can_reach_p[i][j] and can_reach_a[i][j]:
                    ans.append([i, j])
        
        return ans

    def dfs_sea(self, matrix, can_reach, r, c):
        can_reach[r][c] = 1
        direction = [-1, 0, 1, 0, -1]
        m = len(matrix)
        n = len(matrix[0])
        for k in range(4):
            x = r + direction[k]
            y = c + direction[k+1]
            if x >= 0 and x < m and y >=0 and y < n and not can_reach[x][y] and matrix[r][c] <= matrix[x][y]:
                self.dfs_sea(matrix, can_reach, x, y)



class BackTracking(object):
    # 回溯需要记录状态
    # 即路径
    # 1. 按引用传状态；2. 所有的状态递归后回改
    # 需要记录状态的深度有限搜索

    # 排列
    # 修改输出方式
    # 46. Permutation(Medium)
    def permute(self, nums):
        ans = []
        self.backtracking_permute(nums, 0, ans)
        return ans

    def backtracking_permute(self, nums, first, ans):
        if first == len(nums) - 1:
            ans.append(nums.copy()) # python下指向同一数组，需要copy
            return

        for i in range(first, len(nums)):
            nums[first], nums[i] = nums[i], nums[first]     # 修改当前节点状态
            self.backtracking_permute(nums, first+1, ans)   # 递归子节点
            nums[first], nums[i] = nums[i], nums[first]     # 回改当前节点状态


    # 组合
    # 修改输出方式
    # 77. Combination(Medium)
    def combine(self, n, k):
        ans = []
        comb = []
        self.backtracking_combine(ans, comb, 1, n, k)
        return ans

    def backtracking_combine(self, ans, comb, pos, n, k):
        if len(comb) == k:
            ans.append(comb.copy())
            return

        for i in range(pos, n+1):
            comb.append(i)
            self.backtracking_combine(ans, comb, i+1, n, k)
            comb.pop()

    # 另一种写法
    def another_combine(self, n, k):
        nums = [_ for _ in range(1, n+1)]
        ans = []
        comb = []
        self.backtracking_another_combine(nums, 0, k, ans, comb)
        return ans

    def backtracking_another_combine(self, nums, pos, k, ans, comb):
        if len(comb) == k:
            ans.append(comb.copy())
            return

        # 剪枝
        if len(nums) - pos+1 < k - len(comb):
            return  

        for i in range(pos, len(nums)):
            comb.append(nums[i])
            self.backtracking_another_combine(nums, i+1, k, ans, comb)
            comb.pop()
                        

    # 矩阵中搜字符串
    # 修改访问标记
    # 79. Word Search(Medium)
    def exist(self, board, word):
        m = len(board)
        n = len(board[0])
        if (not m) or (not n):
            return False

        visited = [[0]*n for _ in range(m)]
        direction = [-1, 0, 1, 0, -1]
        for i in range(m):
            for j in range(n):
                if self.backtrack_exist(board, word, i, j, visited, 0, direction):
                    return True

        return False

    
    def backtrack_exist(self, board, word, r, c, visited, pos, direction):
        if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]) or visited[r][c]:
            return False

        if word[pos] != board[r][c]:
            return False

        if pos == len(word) - 1:
            return True

        visited[r][c] = 1
        for k in range(4):
            x = r + direction[k]
            y = c + direction[k+1]
            if self.backtrack_exist(board, word, x, y, visited, pos+1, direction):
                return True
        visited[r][c] = 0
        

    # N皇后
    # 修改矩阵状态
    # 51. N-Queens(Hard) 
    def solveNQueens(self, n):
        ans = []
        if n == 0:
            return 0
        
        board = [['.'] * n for _ in range(n)]
        colume = [0] * n
        ldiag = [0] * (2*n - 1)     # 主对角线，横坐标 - 纵坐标的值固定
        rdiag = [0] * (2*n - 1)     # 副对角线，横坐标 + 纵坐标的值固定
        
        self.backtrack_Queens(board, ans, colume, ldiag, rdiag, 0, n)
        for res in ans:
            for i in range(n):
                res[i] = ''.join(res[i])
        return ans

    def backtrack_Queens(self, board, ans, colume, ldiag, rdiag, row, n):
        if row == n:
            ans.append(copy.deepcopy(board))
            return

        for i in range(n):
            if colume[i] or ldiag[n-row+i-1] or rdiag[row+i]:
                continue

            board[row][i] = 'Q'
            colume[i] = 1
            ldiag[n-row+i-1] = 1
            rdiag[row+i] = 1

            self.backtrack_Queens(board, ans, colume, ldiag, rdiag, row+1, n)

            board[row][i] = '.'
            colume[i] = 0
            ldiag[n-row+i-1] = 0
            rdiag[row+i] = 0


from collections import deque
from collections import defaultdict
class BFS(object):
    # 经常用来处理最短路径问题
    
    # 934. Shortest Bridge(Medium)
    def shortestBridge(self, grid):
        # dfs寻找第一个岛，并把所有1赋值为2
        direction = [-1, 0, 1, 0, -1]
        m = len(grid)
        n = len(grid[0])
        visited = [[0]*n for _ in range(m)]
        points = deque()
        flipped = False
        # dfs外循环遍历
        for i in range(m):
            if flipped:
                break
            for j in range(n):
                if grid[i][j]:
                    self.dfs_island(points, grid, visited, m, n, i, j)
                    flipped = True
                    break

        # bfs寻找第二个岛，将海填为2
        level = 0
        while points:
            level += 1
            n_points = len(points)
            while n_points:
                [r, c] = points.popleft()
                grid[r][c] = 2
                for k in range(4):
                    x = r + direction[k]
                    y = c + direction[k+1]
                    if x >= 0 and x < m and y >= 0 and y < n:
                        if grid[x][y] == 2:
                            continue
                        elif grid[x][y] == 1:
                            return level
                        elif grid[x][y] == 0:
                            if not visited[x][y]:
                                visited[x][y] = 1
                                points.append([x, y])
                            
                n_points -= 1 

        return 0

    def dfs_island(self, points, grid, visited, m, n, i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return

        if visited[i][j] == 1 or grid[i][j] == 2:
            return
        else:
            visited[i][j] = 1

        if grid[i][j] == 0:
            # grid[i][j] = 2
            points.append([i, j])
            return
        
        grid[i][j] = 2
        direction = [-1, 0, 1, 0, -1]
        for k in range(4):
            x = i + direction[k]
            y = j + direction[k+1]
            self.dfs_island(points, grid, visited, m, n, x, y)


    # 126. Word Ladder II(Hard)
    # 广度优先搜索，构建图
    # 找最短路径
    # TODO: 超时，待优化
    def findLadders(self, beginWord, endWord, wordList):
        if endWord not in wordList:
            return []

        # 构建图
        graph = defaultdict(list)
        if not beginWord in wordList:
            wordList = [beginWord] + wordList
        n = len(wordList)

        for i in range(n):
            for j in range(i+1, n):
                word_0 = wordList[i]
                word_1 = wordList[j]
                if len(word_0) != len(word_1):
                    break
                
                acount = 0
                for k in range(len(word_0)):
                    if word_0[k] == word_1[k]:
                        acount += 1
                if acount == (len(word_0) - 1):
                    graph[word_0].append(word_1)
                    graph[word_1].append(word_0)
        
        # bfs & backtrack
        ans = []
        search_queue = deque()
        search_queue.append([beginWord])
        shortest_len = float('inf')
        while search_queue:
            path = search_queue.popleft()
            word = path[-1]
            if len(path) <= shortest_len:
                if word == endWord:
                    ans.append(path.copy())
                    shortest_len = len(path)
                else:
                    for node in graph[word]:
                        if not (node in path):
                            search_queue.append(path+[node])

        return ans



        
                    
                      
        

class Solution(object):
    # Basic Excises
    # 130. Surrounded Regions(Medium)
    def solve(self, board):
        m = len(board)
        n = len(board[0])
        connect_to_O = [[0]*n for _ in range(m)]

        for i in range(m):
            if board[i][0] == 'O':
                self.dfs_ox(board, connect_to_O, i, 0)
        for i in range(m):
            if board[i][n-1] == 'O': 
                self.dfs_ox(board, connect_to_O, i, n-1)

        for i in range(n):
            if board[0][i] == 'O':
                self.dfs_ox(board, connect_to_O, 0, i)
        for i in range(n):
            if board[m-1][i] == 'O':
                self.dfs_ox(board, connect_to_O, m-1, i)

        for i in range(1, m-1):
            for j in range(1, n-1):
                if board[i][j] == 'O' and not connect_to_O[i][j]:
                    board[i][j] = 'X'

        return board

    def dfs_ox(self, board, connect_to_O, r, c):
        m = len(board)
        n = len(board[0])
        direction = [-1, 0, 1, 0, -1]
        connect_to_O[r][c] = 1
        for k in range(4):
            x = r + direction[k]
            y = c + direction[k+1]
            if x >= 0 and x < m and y >= 0 and y < n and board[x][y] == 'O' and not connect_to_O[x][y]:
                self.dfs_ox(board, connect_to_O, x, y)


    # Harder Excises
    # 47. Permutation II(Medium)
    def permuteUnique(self, nums):
        ans = []
        nums = sorted(nums)
        visited = [0] * len(nums)
        perm = []
        self.backtracking_permute(perm, nums, ans, visited)
        return ans

    def backtracking_permute(self, perm, nums, ans, visited):
        if len(perm) == len(nums):
            ans.append(perm.copy())
            return

        for i in range(len(nums)):
            if visited[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and visited[i-1] == 0:
                continue
            visited[i] = 1
            self.backtracking_permute(perm + [nums[i]], nums, ans, visited)
            visited[i] = 0


    # Harder Excises
    # 40. Combination Sum II(Medium)
    def combinationSum2(self, candidates, target):
        ans = []
        comb = []
        candidates = sorted(candidates)
        visited = [0] * len(candidates)
        self.backtracking_combine(candidates, target, comb, ans, 0, visited)
        return ans

    def backtracking_combine(self, candidates, target, comb, ans, pos, visited):
        if sum(comb) == target:
            ans.append(comb.copy())
            return

        if pos < len(candidates) and candidates[pos] > target:
            return

        if sum(comb) > target:
            return

        for i in range(pos, len(candidates)):
            if visited[i]:
                continue
            if i > 0 and candidates[i-1] == candidates[i] and visited[i-1] == 0:
                continue
            comb.append(candidates[i])
            visited[i] = 1
            self.backtracking_combine(candidates, target, comb, ans, i+1, visited)
            comb.pop()
            visited[i] = 0


    # 37. Sudoku Solver(Hard)
    def solveSudoku(self, board):
        pass






if __name__ == '__main__':
    # ******************* 深度优先搜索 *******************
    # dfs = DFS()
    
    # 695. Max Area of Island
    # Input = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    #         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #         [0,1,1,0,1,0,0,0,0,0,0,0,0],
    #         [0,1,0,0,1,1,0,0,1,0,1,0,0],
    #         [0,1,0,0,1,1,0,0,1,1,1,0,0],
    #         [0,0,0,0,0,0,0,0,0,0,1,0,0],
    #         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #         [0,0,0,0,0,0,0,1,1,0,0,0,0]]
    # res = dfs.maxAreaOfIsland_stack(Input)
    # res = dfs.maxAreaOfIsland_stack_with_visited_list(Input)
    # res = dfs.maxAreaOfIsland_recuresion_1(Input)

    # 547. Friends Circles
    # Input = [[1,1,0],
    # [1,1,0],
    # [0,0,1]]
    # res = dfs.findCircleNum(Input)
    # res = dfs.findCircleNum_stack(Input)

    # 417. Pacific Atlantic Water Flow(Medium)
    # Input = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    # res = dfs.pacificAtlantic(matrix=Input)


    # ******************* 回溯 *******************
    # backtracking = BackTracking()
    
    # 46. Permutation
    # Input = [1, 2, 3]
    # res = backtracking.permute(Input)

    # 77. Combination
    # n, k = 4, 2
    # res = backtracking.combine(n, k)
    # res = backtracking.another_combine(n, k)

    # 79. Word Search
    # word = "ABCCED"
    # board =[['A','B','C','E'], 
    # ['S','F','C','S'],
    # ['A','D','E','E']]
    # # board = [["b"],["a"],["b"]]
    # # word = "bbabab"
    # res = backtracking.exist(board, word)

    # 51. N-Queens(Hard)
    # n = 4
    # res = backtracking.solveNQueens(n)

    # ******************* 广度优先搜索 *******************
    bfs = BFS()

    # 934. Shortest Bridge
    # Input=[[1,1,1,1,1],
    # [1,0,0,0,1],
    # [1,0,1,0,1],
    # [1,0,0,0,1],
    # [1,1,1,1,1]]
    # Input = [[0,1,0],[0,0,0],[0,0,1]]
    # res = bfs.shortestBridge(Input)

    # 126. Word Ladders
    # beginWord = "hit"
    # endWord = "cog"
    # wordList = ["hot","dot","dog","lot","log","cog"]
    # beginWord = "leet"
    # endWord = "code"
    # wordList = ["lest","leet","lose","code","lode","robe","lost"]
    beginWord = "cet"
    endWord = "ism"
    wordList = ["kid","tag","pup","ail","tun","woo","erg","luz","brr","gay","sip","kay","per","val","mes","ohs","now","boa","cet","pal","bar","die","war","hay","eco","pub","lob","rue","fry","lit","rex","jan","cot","bid","ali","pay","col","gum","ger","row","won","dan","rum","fad","tut","sag","yip","sui","ark","has","zip","fez","own","ump","dis","ads","max","jaw","out","btu","ana","gap","cry","led","abe","box","ore","pig","fie","toy","fat","cal","lie","noh","sew","ono","tam","flu","mgm","ply","awe","pry","tit","tie","yet","too","tax","jim","san","pan","map","ski","ova","wed","non","wac","nut","why","bye","lye","oct","old","fin","feb","chi","sap","owl","log","tod","dot","bow","fob","for","joe","ivy","fan","age","fax","hip","jib","mel","hus","sob","ifs","tab","ara","dab","jag","jar","arm","lot","tom","sax","tex","yum","pei","wen","wry","ire","irk","far","mew","wit","doe","gas","rte","ian","pot","ask","wag","hag","amy","nag","ron","soy","gin","don","tug","fay","vic","boo","nam","ave","buy","sop","but","orb","fen","paw","his","sub","bob","yea","oft","inn","rod","yam","pew","web","hod","hun","gyp","wei","wis","rob","gad","pie","mon","dog","bib","rub","ere","dig","era","cat","fox","bee","mod","day","apr","vie","nev","jam","pam","new","aye","ani","and","ibm","yap","can","pyx","tar","kin","fog","hum","pip","cup","dye","lyx","jog","nun","par","wan","fey","bus","oak","bad","ats","set","qom","vat","eat","pus","rev","axe","ion","six","ila","lao","mom","mas","pro","few","opt","poe","art","ash","oar","cap","lop","may","shy","rid","bat","sum","rim","fee","bmw","sky","maj","hue","thy","ava","rap","den","fla","auk","cox","ibo","hey","saw","vim","sec","ltd","you","its","tat","dew","eva","tog","ram","let","see","zit","maw","nix","ate","gig","rep","owe","ind","hog","eve","sam","zoo","any","dow","cod","bed","vet","ham","sis","hex","via","fir","nod","mao","aug","mum","hoe","bah","hal","keg","hew","zed","tow","gog","ass","dem","who","bet","gos","son","ear","spy","kit","boy","due","sen","oaf","mix","hep","fur","ada","bin","nil","mia","ewe","hit","fix","sad","rib","eye","hop","haw","wax","mid","tad","ken","wad","rye","pap","bog","gut","ito","woe","our","ado","sin","mad","ray","hon","roy","dip","hen","iva","lug","asp","hui","yak","bay","poi","yep","bun","try","lad","elm","nat","wyo","gym","dug","toe","dee","wig","sly","rip","geo","cog","pas","zen","odd","nan","lay","pod","fit","hem","joy","bum","rio","yon","dec","leg","put","sue","dim","pet","yaw","nub","bit","bur","sid","sun","oil","red","doc","moe","caw","eel","dix","cub","end","gem","off","yew","hug","pop","tub","sgt","lid","pun","ton","sol","din","yup","jab","pea","bug","gag","mil","jig","hub","low","did","tin","get","gte","sox","lei","mig","fig","lon","use","ban","flo","nov","jut","bag","mir","sty","lap","two","ins","con","ant","net","tux","ode","stu","mug","cad","nap","gun","fop","tot","sow","sal","sic","ted","wot","del","imp","cob","way","ann","tan","mci","job","wet","ism","err","him","all","pad","hah","hie","aim","ike","jed","ego","mac","baa","min","com","ill","was","cab","ago","ina","big","ilk","gal","tap","duh","ola","ran","lab","top","gob","hot","ora","tia","kip","han","met","hut","she","sac","fed","goo","tee","ell","not","act","gil","rut","ala","ape","rig","cid","god","duo","lin","aid","gel","awl","lag","elf","liz","ref","aha","fib","oho","tho","her","nor","ace","adz","fun","ned","coo","win","tao","coy","van","man","pit","guy","foe","hid","mai","sup","jay","hob","mow","jot","are","pol","arc","lax","aft","alb","len","air","pug","pox","vow","got","meg","zoe","amp","ale","bud","gee","pin","dun","pat","ten","mob"]
    res = bfs.findLadders(beginWord, endWord, wordList)

    # ******************* 练习 *******************
    # s = Solution()

    # 130. Surrounded Regions(Medium)
    # board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    # board = [["O","O"],["O","O"]]
    # board = [["O","X","X","X","X","X","O","O"],["O","O","O","X","X","X","X","O"],["X","X","X","X","O","O","O","O"],["X","O","X","O","O","X","X","X"],["O","X","O","X","X","X","O","O"],["O","X","X","O","O","X","X","O"],["O","X","O","X","X","X","O","O"],["O","X","X","X","X","O","X","X"]]
    # res = s.solve(board)

    # 47. Permutation II(Medium)
    # nums = [1, 1, 2]
    # res = s.permuteUnique(nums)

    # 40. Combination Sum II(Medium)
    # candidates = [10, 1, 2, 7, 6, 1, 5]
    # # target = 8
    # candidates = [14,6,25,9,30,20,33,34,28,30,16,12,31,9,9,12,34,16,25,32,8,7,30,12,33,20,21,29,24,17,27,34,11,17,30,6,32,21,27,17,16,8,24,12,12,28,11,33,10,32,22,13,34,18,12]
    # target = 27
    # res = s.combinationSum2(candidates, target)





    print(res)
                