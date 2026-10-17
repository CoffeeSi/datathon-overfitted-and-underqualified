import { ReviewWorkspace } from "@/features/review/review-workspace";
export default async function ReviewPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return <ReviewWorkspace key={jobId} jobId={jobId} />;
}
