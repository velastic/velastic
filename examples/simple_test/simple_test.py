from velastic.manager import IterManager
import time
import os
iter_mngr = IterManager(velastic_dir="./examples/.velastic")

# train
while True:
    iter_mngr.iter()
    print(iter_mngr.iter_count)
    time.sleep(1)
    if iter_mngr.should_save_ckpt():
        print("save!")
    if float(os.environ.get("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"))<time.time():
        break
