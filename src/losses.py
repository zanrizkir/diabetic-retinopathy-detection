import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="dr_project")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss untuk klasifikasi multi-kelas. Berbeda dari cross-entropy
    biasa, loss ini secara otomatis MENGECILKAN kontribusi contoh yang
    sudah "mudah" (prediksi sudah confident dan benar) dan MEMBESARKAN
    kontribusi contoh yang "sulit" (model masih ragu/salah) -- ini
    membantu model tidak "puas diri" hanya karena sudah bagus di kelas
    mayoritas (No DR), dan tetap dipaksa belajar dari kelas minoritas
    (Severe/Proliferative) yang secara alami lebih sering salah tebak.

    Dipakai BERSAMA class_weight (bukan pengganti) -- class_weight sudah
    menangani ketimpangan JUMLAH sampel per kelas, sedangkan focal loss
    menangani tingkat KESULITAN per-contoh.
    """

    def __init__(self, gamma: float = 2.0, name: str = "categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulating_factor = tf.pow(1.0 - y_pred, self.gamma)
        loss = modulating_factor * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config