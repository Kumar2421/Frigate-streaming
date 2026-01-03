import { cn } from "@/lib/utils";
import logoImage from "@/logo.png";

type LogoProps = {
  className?: string;
};

export default function Logo({ className }: LogoProps) {
  return (
    <img
      src={logoImage}
      alt="Fusion Nvr"
      className={cn("object-contain", className)}
    />
  );
}


