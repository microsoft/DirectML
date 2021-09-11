import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.disable_v2_behavior()
from MHA import MultiHeadSelfAttention

def Rescale(input, scale, offset=0):
    """Rescaling helper function to scale image elements down to the range [0,1]"""
    dtype = tf.float32
    scale = tf.cast(scale, dtype)
    offset = tf.cast(offset, dtype)
    return tf.cast(input, dtype) * scale + offset

def gelu(x):
    """ The GELU Activation function: defined as x*CDF(x) for the Standard Normal(0,1) Distribution"""

    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


def MLP(hidden_dim, embed_dim, rate=0.2):
    model=tf.keras.Sequential(
            [   tf.keras.layers.Dense(hidden_dim, activation=gelu),
                tf.keras.layers.Dropout(rate),
                tf.keras.layers.Dense(embed_dim, activation=gelu)
            ]
        )
    return model

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.mlp = MLP(mlp_hidden_dim, embed_dim,0.2)
        self.MHA_layer = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)

    def call(self, input_embeddings, training=True):
        input_embeddings_norm = self.layernorm1(input_embeddings)
        output = self.MHA_layer(input_embeddings_norm)
        output = self.dropout1(output, training=training)
        output_1 = output + input_embeddings
        #Skip Connection: Adding input_embeddings to the output 

        output_norm = self.layernorm2(output_1)
        MLP_output = self.mlp(output_norm)
        MLP_output = self.dropout2(MLP_output, training=training)
        return MLP_output + output_1 
        #Skip Connection: Adding output_1 to the final output MLP_output 

class PatchExtractEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, patch_embedding_dim, patch_size, patch_stride):
        super(PatchExtractEncoder, self).__init__()
        self.patch_size= patch_size
        self.patch_stride= patch_stride
        self.num_patches = num_patches
        self.classification_emb = self.add_weight("class_emb", shape=(1, 1, patch_embedding_dim))
        self.projection = tf.keras.layers.Dense(patch_embedding_dim)
        self.patch_embedding_dim=patch_embedding_dim
        #We define learnable Position embedding weights that are added to the path embeddings
        self.position_embeddings = self.add_weight("pos_emb", shape=(1, num_patches + 1, patch_embedding_dim))
    
    def get_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = self.get_patches(images)
        
        classification_emb = tf.broadcast_to( self.classification_emb, [batch_size, 1, self.patch_embedding_dim])
        
        proj_patches = self.projection(patches) 
        proj_patches = tf.concat([classification_emb, proj_patches], axis=1)
        proj_patches += self.position_embeddings
        return proj_patches

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        patch_stride,
        num_layers,
        num_classes,
        embedding_dim,
        num_heads,
        mlp_hidden_dim,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        #The number of patches is analagous to the number of words in a sequence being fed to a transformer. The image patches are flattened and transformed to a lower dimensional embedding space (embedding dim) 
        self.patch_dim = (patch_size ** 2) * 3
        #Flatting the path results in a path_dim dimensional vector. For patch_size =4, this is 4*4*3 = 48 dimensional
        self.embedding_dim = embedding_dim
        self.num_stacked_layers = num_layers
        #Adding learnable classification embedding weights to the model class

        self.PatchExtractEncoder= PatchExtractEncoder(num_patches,embedding_dim, patch_size, patch_stride)
        self.transformer_layers = [TransformerEncoderBlock(embedding_dim, num_heads, mlp_hidden_dim) for i in range(self.num_stacked_layers)]
        self.layernorm= tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation=gelu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation=gelu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        self.flatten= tf.keras.layers.Flatten()
        self.dropout= tf.keras.layers.Dropout(0.25)

    def call(self, images, training=True):
        images = Rescale(images, 1.0 / 255.0)
        #Image elements are scaled by 1/255 so that each element of the image x is now between 0 and 1 

         #Extract patches using specified patch_size and patch_stride parameters, and return flatten patches of shape [batch_size, number of patches, self.patch_dim], Flattened Patches are projected down to (embedding_dim) sized embeddings
        x = self.PatchExtractEncoder(images)

        for transformer_encoder_block in self.transformer_layers:
            x = transformer_encoder_block(x, training)

        x = self.layernorm(x)
        res = self.classifier(x[:, 0])
        return res
