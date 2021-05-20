from phe import paillier
from phe.util import powmod, invert


def parallelization_encryption(public_key, unencrypted_list, prec, elem):
    
    return public_key.encrypt(float(unencrypted_list[elem]), precision=prec)



def parallelization_encryption_rvalues(public_key, unencrypted_list, prec, r_values, elem):
    
    return public_key.encrypt(float(unencrypted_list[elem]), precision=prec, r_value=r_values[elem])



def parallelization_encryption_int(public_key, unencrypted_list, prec, r_values, elem):
    
    return public_key.encrypt(unencrypted_list[elem], precision=prec, r_value=r_values[elem])
    
    
    
def parallelization_decryption(private_key, encrypted_list, elem):
    
    return private_key.decrypt(encrypted_list[elem])



def transform_encrypted_domain(public_key_origin, public_key_destination, weights_origin, precision, encrypted_Xi, encrypted_Xi_dest, x):
    aux = (encrypted_Xi[x].ciphertext(be_secure=False) * invert(weights_origin[x].ciphertext(be_secure=False), public_key_origin.nsquare))
    D = (((aux - 1) %public_key_origin.nsquare) // public_key_origin.n) %public_key_origin.n
    
    # Treat negative numbers
    if D > public_key_origin.max_int:
        D = D - public_key_origin.n
    
    # Reescale to convert back from int to float
    D = D / pow(16, -weights_origin[x].exponent)

    D_transformed = public_key_destination.encrypt(D, precision=precision) # Use public key of DON N-1 to encrypt
    D_transformed = encrypted_Xi_dest[x] - D_transformed # Remove the noise Xi (substraction in the encrypted domain of DON destination)
    
    return D_transformed