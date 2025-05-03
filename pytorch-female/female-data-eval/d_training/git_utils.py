from git import Repo


# Git functions
def create_git_tag(repo, tag_name):
    try:
        tag = repo.create_tag(tag_name)
        return tag
    except Exception as e:
        print(f"Error creating Git tag: {e}")
        return None

def get_git_hash(repo):
    try:
        return repo.head.commit.hexsha
    except Exception as e:
        print(f"Error getting Git hash: {e}")
        return None