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

        if isinstance(nums, list):

            sort_nums = nums.copy()
            sort_nums.sort()
            # print(nums.index(sort_nums[-1]))
            if nums.__len__() != 1:
                max_index = nums.index(sort_nums[-1])
                tree_node = TreeNode(max_index)

                tree_node.left = nums[0:max_index]
                tree_node.right = nums[max_index + 1: nums.__len__()]
                # print(tree_node.left, tree_node.right)
                self.constructMaximumBinaryTree(tree_node.left)
                self.constructMaximumBinaryTree(tree_node.right)
                if tree_node.left.__len__() != 1:
                    return tree_node.left
                if tree_node.right.__len__() != 1:
                    return tree_node.right
            else:
                return tree_node

solution = Solution()
result = solution.constructMaximumBinaryTree([3, 2, 1, 6, 0, 5])
print(result)
