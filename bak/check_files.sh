# 1. 檢查目錄是否存在
ls -la /Users/xrickliao/WorkSpaces/DataSets/speechocean762/

# 2. 檢查是否有音檔
find /Users/xrickliao/WorkSpaces/DataSets/speechocean762/ -name "*.wav" | head -10

# 3. 檢查所有檔案類型
find /Users/xrickliao/WorkSpaces/DataSets/speechocean762/ -type f | head -20

# 4. 檢查目錄結構
tree /Users/xrickliao/WorkSpaces/DataSets/speechocean762/ -L 3