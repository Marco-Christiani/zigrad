import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import { Zap, Lock, Cpu, Scale, LucideIcon } from 'lucide-react';

type FeatureItem = {
  title: string;
  Icon: LucideIcon,
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Blazing Fast Performance',
    Icon: Zap,
    description: (
      <>
        {/* Achieve 2.5x+ speedup over compiled PyTorch models on Apple Silicon. */}
        Zigrad's zero-overhead abstractions and careful memory management deliver
        exceptional performance across architectures. Faster than PyTorch, libtorch,
        Tensorflow, and Tinygrad.
        {/* Zigrad is the fastest deep learning framework we have ever tested. */}
      </>
    ),
  },
  {
    title: 'Systems-Level Control',
    Icon: Lock,
    description: (
      <>
        Fine-grained control over memory management and performance characteristics.
        Optimize for your specific hardware and requirements without fighting
        abstraction layers.
      </>
    ),
  },
  {
    title: 'Efficient Resource Usage',
    Icon: Cpu,
    description: (
      <>
        Minimal and transparent heap allocations.
        Hardware-optimized implementations leveraging BLAS, SIMD, and platform-specific
        accelerators.
      </>
    ),
  },
  {
    title: 'Production Ready',
    Icon: Scale,
    description: (
      <>
        Built for real-world deep learning deployment. {/*Seamless integration with
        existing ML pipelines via ONNX support.*/} Comprehensive test coverage and
        benchmarking suite. More production-specific features in development.
      </>
    ),
  },
];

function Feature({ title, Icon, description }: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center">
        <Icon className={styles.featureSvg} role="img" size={64} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center margin-bottom--xl">
          <Heading as="h2" className="margin-bottom--md">
            Deep Learning Framework Built for Performance
          </Heading>
          <p className={clsx('hero__subtitle', 'margin-bottom--lg')}>
            Zigrad combines the ergonomics of high-level ML frameworks with
            the performance and control of systems programming.
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
