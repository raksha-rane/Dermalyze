/**
 * Client-side image encryption for privacy.
 *
 * Security model:
 * - Deterministic AES-256-GCM key derived from the user's Supabase UUID via
 *   PBKDF2 (100 000 iterations, SHA-256). Because the key is derived from a
 *   value the user always has post-authentication, it works across every
 *   device, browser, and session — no local key storage required.
 * - Images in Supabase Storage are additionally protected by RLS + signed
 *   URLs, so an attacker would need both authenticated access AND the userId
 *   to reach and decrypt the ciphertext.
 *
 * Wire format: [12-byte IV][AES-GCM ciphertext + 16-byte auth tag]
 *
 * Legacy note:
 * - v2 files (magic header "DLZ2") used a random device-bound key stored in
 *   localStorage. Decryption for those is still attempted if the local key
 *   exists, but new uploads always use the deterministic derivation so key
 *   loss is impossible.
 */

const AES_GCM_IV_BYTES = 12; // standard IV size for AES-GCM
const ENCRYPTION_MAGIC_V2 = new Uint8Array([0x44, 0x4c, 0x5a, 0x32]); // "DLZ2"

// ── v2 legacy helpers (read-only — never create new v2 keys) ────────────────

const AES_KEY_BYTES = 32;
const KEY_STORAGE_PREFIX_V2 = 'dermalyze:image-key:v2:';

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
    const encoded = window.localStorage.getItem(`${KEY_STORAGE_PREFIX_V2}${userId}`);
    if (!encoded) return null;
    const keyBytes = base64ToBytes(encoded);
    if (keyBytes.byteLength !== AES_KEY_BYTES) return null;
    return keyBytes;
  } catch {
    return null;
  }
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

// ── Deterministic key derivation (used for all new encryptions) ─────────────

function assertValidUserId(userId: string): void {
  if (!userId || !userId.trim()) {
    throw new Error('Missing user context for image encryption.');
  }
}

/**
 * Derives a deterministic AES-256-GCM key from the authenticated user's ID.
 *
 * Because the userId (a Supabase UUID) is always available after login, this
 * key can be reconstructed on any device/browser without storing anything
 * locally — eliminating the key-loss problem entirely.
 */
async function deriveDeterministicKey(userId: string): Promise<CryptoKey> {
  assertValidUserId(userId);
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

// ── v2 payload detection ────────────────────────────────────────────────────

function isV2EncryptedPayload(data: Uint8Array): boolean {
  if (data.byteLength < ENCRYPTION_MAGIC_V2.length + AES_GCM_IV_BYTES + 16) {
    return false;
  }
  for (let i = 0; i < ENCRYPTION_MAGIC_V2.length; i++) {
    if (data[i] !== ENCRYPTION_MAGIC_V2[i]) return false;
  }
  return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Encrypts an image blob using AES-256-GCM with a deterministic key.
 *
 * Output format: [12-byte IV][ciphertext + 16-byte auth tag]
 *
 * @param imageBlob - Original image as Blob (WebP, JPEG, PNG, etc.)
 * @param userId - User's Supabase UUID for key derivation
 * @returns Encrypted blob
 */
export async function encryptImage(imageBlob: Blob, userId: string): Promise<Blob> {
  try {
    const key = await deriveDeterministicKey(userId);

    // Generate random IV (12 bytes is standard for AES-GCM)
    const iv = crypto.getRandomValues(new Uint8Array(AES_GCM_IV_BYTES));

    // Read image as ArrayBuffer
    const imageData = await imageBlob.arrayBuffer();

    // Encrypt the image data
    const encryptedData = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, imageData);

    // Format: [12-byte IV][ciphertext+auth-tag]  (no v2 magic header)
    const combined = new Uint8Array(iv.length + encryptedData.byteLength);
    combined.set(iv, 0);
    combined.set(new Uint8Array(encryptedData), iv.length);

    return new Blob([combined], { type: 'application/octet-stream' });
  } catch (err) {
    console.error('Image encryption failed:', err);
    throw new Error('Failed to encrypt image. Please try again.');
  }
}

/**
 * Decrypts an encrypted image blob.
 *
 * Handles three cases in order:
 *  1. v2 payload (DLZ2 header) — tries the device-local key if it still exists.
 *  2. v1 / deterministic payload — derives key from userId (always works).
 *
 * @param encryptedBlob - Encrypted blob
 * @param userId - User's Supabase UUID for key derivation
 * @param originalMimeType - MIME type for the decrypted output
 * @returns Decrypted image as Blob
 */
export async function decryptImage(
  encryptedBlob: Blob,
  userId: string,
  originalMimeType: string = 'image/webp'
): Promise<Blob> {
  try {
    const encryptedData = await encryptedBlob.arrayBuffer();
    const dataView = new Uint8Array(encryptedData);

    let decryptedData: ArrayBuffer;

    if (isV2EncryptedPayload(dataView)) {
      // v2 payload — attempt with locally-stored key (may not exist).
      const localKeyBytes = readStoredV2Key(userId);
      if (!localKeyBytes) {
        throw new Error(
          'This image was encrypted with a device-specific key that is no longer available. ' +
            'It cannot be decrypted on this device.'
        );
      }
      const key = await importAesGcmKey(localKeyBytes);
      const ivStart = ENCRYPTION_MAGIC_V2.length;
      const ivEnd = ivStart + AES_GCM_IV_BYTES;
      const iv = dataView.slice(ivStart, ivEnd);
      const ciphertext = dataView.slice(ivEnd);
      decryptedData = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
    } else {
      // Deterministic / v1 payload: [12-byte IV][ciphertext+auth-tag]
      const key = await deriveDeterministicKey(userId);
      const iv = dataView.slice(0, AES_GCM_IV_BYTES);
      const ciphertext = dataView.slice(AES_GCM_IV_BYTES);
      decryptedData = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
    }

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
