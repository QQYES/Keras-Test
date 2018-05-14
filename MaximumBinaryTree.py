class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        # tree_node = TreeNode()
        sort_nums = nums.copy()
        sort_nums.sort()
        # print(nums.index(sort_nums[-1]))

        max_index = nums.index(sort_nums[-1])
        tree_node = self.sub_function(TreeNode(nums[max_index]), nums)
        return tree_node

    def sub_function(self, tree_node, nums):
        if isinstance(nums, list):
            if nums.__len__() == 1:
                return tree_node[nums[0]]
            sort_nums = nums.copy()
            sort_nums.sort()
            # print(nums.index(sort_nums[-1]))

            max_index = nums.index(sort_nums[-1])
            tree_node = TreeNode(nums[max_index])
            left_nums = nums[0:max_index]
            right_nums = nums[max_index + 1: nums.__len__()]

            if left_nums.__len__() != 0:
                return self.sub_function(tree_node.left, left_nums)
            if right_nums.__len__() != 0:
                return self.sub_function(tree_node.right, right_nums)


solution = Solution()
result = solution.constructMaximumBinaryTree([3, 2, 1, 6, 0, 5])
print(result.val, result.left, result.right)
