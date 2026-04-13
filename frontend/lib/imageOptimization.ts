/**
 * Image optimization utilities for efficient storage in Supabase.
 *
 * Converts images to WebP format with compression to reduce storage space
 * while maintaining fast retrieval. WebP provides ~30-50% better compression
 * than JPEG/PNG with similar visual quality.
 */

export interface OptimizationOptions {
  /** Max width/height in pixels. Default: 1024 for full images */
  maxDimension?: number;
  /** WebP quality (0-1). Default: 0.75 for good balance of size/quality */
  quality?: number;
  /** Output format. Default: 'image/webp' */
  format?: 'image/webp' | 'image/jpeg' | 'image/png';
}

/**
 * Optimizes an image for efficient storage.
 *
 * @param dataUrl - Base64 data URL of the source image
 * @param options - Optimization configuration
 * @returns Promise resolving to optimized image as Blob
 */
export async function optimizeImage(
  dataUrl: string,
  options: OptimizationOptions = {}
): Promise<Blob> {
  const { maxDimension = 1024, quality = 0.75, format = 'image/webp' } = options;

  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      try {
        // Calculate scaled dimensions maintaining aspect ratio
        const scale = Math.min(1, maxDimension / Math.max(img.width, img.height));
        const targetWidth = Math.round(img.width * scale);
        const targetHeight = Math.round(img.height * scale);

        // Create canvas and draw scaled image
        const canvas = document.createElement('canvas');
        canvas.width = targetWidth;
        canvas.height = targetHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        // Use high-quality image smoothing for better downscaling
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

        // Convert to blob with specified format and quality
        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Canvas toBlob returned null'));
            }
          },
          format,
          quality
        );
      } catch (err) {
        reject(err);
      }
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = dataUrl;
  });
}

/**
 * Generates a thumbnail version of an image.
 * Uses a smaller max dimension and can use lower quality for lists/previews.
 *
 * @param dataUrl - Base64 data URL of the source image
 * @param maxDimension - Max width/height in pixels. Default: 200
 * @returns Promise resolving to thumbnail as Blob
 */
export async function generateThumbnail(
  dataUrl: string,
  maxDimension: number = 200
): Promise<Blob> {
  return optimizeImage(dataUrl, {
    maxDimension,
    quality: 0.7,
    format: 'image/webp',
  });
}

/**
 * Calculates the file extension for optimized images.
 * Use this to ensure correct file extension when uploading to storage.
 */
export function getOptimizedExtension(format: string = 'image/webp'): string {
  const extensionMap: Record<string, string> = {
    'image/webp': 'webp',
    'image/jpeg': 'jpg',
    'image/png': 'png',
  };
  return extensionMap[format] || 'webp';
}

/**
 * Estimates storage savings from WebP conversion.
 * WebP typically provides 30-50% reduction compared to JPEG/PNG.
 */
export function estimateWebPSavings(originalSizeBytes: number): {
  estimatedSize: number;
  savingsPercent: number;
} {
  const savingsPercent = 40; // Conservative estimate
  const estimatedSize = Math.round(originalSizeBytes * (1 - savingsPercent / 100));
  return { estimatedSize, savingsPercent };
}
