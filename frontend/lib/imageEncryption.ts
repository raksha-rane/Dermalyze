/**
 * Client-side image encryption for privacy.
 *
 * Security model:
 * - v2 (current): uses a random 256-bit key generated on the client and stored
 *   locally per user/device. The server never receives this key.
 * - v1 (legacy): deterministic key derived from userId. Kept only for backward
 *   compatibility to decrypt already-stored historical files.
 *
 * Note: because v2 keys are device-bound, encrypted images are readable only on
 * devices that hold the local key.
 */

const AES_KEY_BYTES = 32; // 256-bit key
const AES_GCM_IV_BYTES = 12; // standard IV size for AES-GCM
const ENCRYPTION_MAGIC_V2 = new Uint8Array([0x44, 0x4c, 0x5a, 0x32]); // "DLZ2"
const KEY_STORAGE_PREFIX_V2 = 'dermalyze:image-key:v2:';

function assertValidUserId(userId: string): void {
  if (!userId || !userId.trim()) {
    throw new Error('Missing user context for image encryption.');
  }
}

function getUserKeyStorageKey(userId: string): string {
  return `${KEY_STORAGE_PREFIX_V2}${userId}`;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

function readStoredV2Key(userId: string): Uint8Array | null {
  try {
    const encoded = window.localStorage.getItem(getUserKeyStorageKey(userId));
    if (!encoded) return null;
    const keyBytes = base64ToBytes(encoded);
    if (keyBytes.byteLength !== AES_KEY_BYTES) return null;
    return keyBytes;
  } catch {
    return null;
  }
}

function storeV2Key(userId: string, keyBytes: Uint8Array): void {
  window.localStorage.setItem(getUserKeyStorageKey(userId), bytesToBase64(keyBytes));
}

async function importAesGcmKey(keyBytes: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    keyBytes,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

async function getOrCreateV2Key(userId: string): Promise<CryptoKey> {
  assertValidUserId(userId);

  let keyBytes = readStoredV2Key(userId);
  if (!keyBytes) {
    keyBytes = crypto.getRandomValues(new Uint8Array(AES_KEY_BYTES));
    storeV2Key(userId, keyBytes);
  }
  return importAesGcmKey(keyBytes);
}

async function getExistingV2Key(userId: string): Promise<CryptoKey> {
  assertValidUserId(userId);
  const keyBytes = readStoredV2Key(userId);
  if (!keyBytes) {
    throw new Error(
      'Encrypted image key not found on this device. Historical images cannot be decrypted here.'
    );
  }
  return importAesGcmKey(keyBytes);
}

async function deriveLegacyV1Key(userId: string): Promise<CryptoKey> {
  const keyMaterial = new TextEncoder().encode(userId);
  const baseKey = await crypto.subtle.importKey('raw', keyMaterial, { name: 'PBKDF2' }, false, [
    'deriveKey',
  ]);
  const salt = new TextEncoder().encode(`dermalyze-salt-${userId}`);
  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    baseKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

function isV2EncryptedPayload(data: Uint8Array): boolean {
  if (data.byteLength < ENCRYPTION_MAGIC_V2.length + AES_GCM_IV_BYTES + 16) {
    return false;
  }
  for (let i = 0; i < ENCRYPTION_MAGIC_V2.length; i++) {
    if (data[i] !== ENCRYPTION_MAGIC_V2[i]) return false;
  }
  return true;
}

/**
 * Encrypts an image blob using AES-GCM.
 *
 * The encrypted output includes:
 * - 12-byte IV (initialization vector) prepended to the data
 * - Encrypted image data
 * - 16-byte authentication tag (built into AES-GCM)
 *
 * @param imageBlob - Original image as Blob (WebP, JPEG, PNG, etc.)
 * @param userId - User's ID for key derivation
 * @returns Encrypted blob (IV + encrypted data + auth tag)
 */
export async function encryptImage(imageBlob: Blob, userId: string): Promise<Blob> {
  try {
    // v2 encryption uses a random, device-bound key (not userId-derived).
    const key = await getOrCreateV2Key(userId);

    // Generate random IV (12 bytes is standard for AES-GCM)
    const iv = crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));

    // Read image as ArrayBuffer
    const imageData = await imageBlob.arrayBuffer();

    // Encrypt the image data
    const encryptedData = await crypto.subtle.encrypt(
      {
        name: 'AES-GCM',
        iv: iv,
      },
      key,
      imageData
    );

    // v2 format: [4-byte magic "DLZ2"][12-byte IV][ciphertext+auth-tag]
    const combinedData = new Uint8Array(
      ENCRYPTION_MAGIC_V2.length + iv.length + encryptedData.byteLength
    );
    combinedData.set(ENCRYPTION_MAGIC_V2, 0);
    combinedData.set(iv, ENCRYPTION_MAGIC_V2.length);
    combinedData.set(new Uint8Array(encryptedData), ENCRYPTION_MAGIC_V2.length + iv.length);

    return new Blob([combinedData], { type: 'application/octet-stream' });
  } catch (err) {
    console.error('Image encryption failed:', err);
    throw new Error('Failed to encrypt image. Please try again.');
  }
}

/**
 * Decrypts an encrypted image blob.
 *
 * @param encryptedBlob - Encrypted blob (IV + encrypted data + auth tag)
 * @param userId - User's ID for key derivation
 * @param originalMimeType - Original image MIME type (e.g., 'image/webp')
 * @returns Decrypted image as Blob
 */
export async function decryptImage(
  encryptedBlob: Blob,
  userId: string,
  originalMimeType: string = 'image/webp'
): Promise<Blob> {
  try {
    // Read encrypted data
    const encryptedData = await encryptedBlob.arrayBuffer();
    const dataView = new Uint8Array(encryptedData);

    let decryptedData: ArrayBuffer;

    if (isV2EncryptedPayload(dataView)) {
      // v2 payload: key is device-bound and stored locally.
      const key = await getExistingV2Key(userId);
      const ivStart = ENCRYPTION_MAGIC_V2.length;
      const ivEnd = ivStart + AES_GCM_IV_BYTES;
      const iv = dataView.slice(ivStart, ivEnd);
      const ciphertext = dataView.slice(ivEnd);
      decryptedData = await crypto.subtle.decrypt(
        {
          name: 'AES-GCM',
          iv,
        },
        key,
        ciphertext
      );
    } else {
      // Legacy v1 payload fallback for historical .enc files:
      // format was [12-byte IV][ciphertext+auth-tag].
      const key = await deriveLegacyV1Key(userId);
      const iv = dataView.slice(0, AES_GCM_IV_BYTES);
      const ciphertext = dataView.slice(AES_GCM_IV_BYTES);
      decryptedData = await crypto.subtle.decrypt(
        {
          name: 'AES-GCM',
          iv,
        },
        key,
        ciphertext
      );
    }

    // Return as blob with original MIME type
    return new Blob([decryptedData], { type: originalMimeType });
  } catch (err) {
    console.error('Image decryption failed:', err);
    throw new Error('Failed to decrypt image. The image may be corrupted.');
  }
}

/**
 * Converts a Blob to a data URL for display in <img> tags.
 *
 * @param blob - Image blob
 * @returns Promise resolving to data URL string
 */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Helper to get file extension for encrypted files.
 * Encrypted files are stored with .enc extension.
 */
export function getEncryptedExtension(): string {
  return 'enc';
}
