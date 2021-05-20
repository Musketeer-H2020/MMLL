from phe import paillier


def parallelization_encryption(public_key, unencrypted_list, elem):
    
    return public_key.encrypt(float(unencrypted_list[elem]))
    
    
    
def parallelization_decryption(private_key, encrypted_list, elem):
    
    return private_key.decrypt(encrypted_list[elem])