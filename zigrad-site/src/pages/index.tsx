import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import clsx from 'clsx';
import { Zap, ChartLine } from 'lucide-react';

import styles from './index.module.css';


function DemoVideo(): JSX.Element {
  return (
    <div className="container margin-bottom--lg">
      <div className="video-container">
        <video
          controls
          muted
          loop
          playsInline
          style={{
            width: '100%',
            maxWidth: '800px',
            height: 'auto',
            display: 'block',
            margin: '0 auto',
          }}
        >
          <source
            src="https://github.com/Marco-Christiani/zigrad/raw/refs/heads/main/docs/zigrad-demo.mp4"
            type="video/mp4"
          />
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
}


function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)} data-theme="dark">
      <div className="container">
        <Heading as="h1" className={styles.heroTitleText}>
          {siteConfig.title}
        </Heading>
        <p className={clsx("hero__subtitle", styles.heroProjectTagline)}>{siteConfig.tagline}</p>
        <div className={styles.indexCtas}>
          <Link
            className="button button--primary"
            to="/docs/intro">
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              Get Started <Zap size={24} />
            </div>
          </Link>
          <Link
            className="button button--info"
            to="/docs/benchmarks">
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              View Benchmarks <ChartLine size={24} />
            </div>
          </Link>
          <span className={styles.indexCtasGitHubButtonWrapper}>
            <iframe
              className={styles.indexCtasGitHubButton}
              src="https://ghbtns.com/github-btn.html?user=Marco-Christiani&amp;repo=Zigrad&amp;type=star&amp;count=true&amp;size=large"
              width={120}
              height={30}
              // width={80}
              // height={20}
              title="GitHub Stars"
            />
          </span>
        </div>
        <div className="margin-top--md margin-bottom--lg">
          <CodeBlock language='sh'>
            {`git clone https://github.com/Marco-Christiani/zigrad
cd zigrad/examples/mnist && make`}
          </CodeBlock>
        </div>
        <DemoVideo></DemoVideo>

      </div>
    </header>
  );
}


function CodeSection() {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <Heading as="h2" className="text--center margin-bottom--lg">
          Write High Performance Deep Learning Code
        </Heading>
        <div className="row">
          <div className="col col--6">
            <CodeBlock language="zig">
              {`// Define a simple model
var model = try Model(f32).init(alloc);
defer model.deinit();

// Add layers
var conv1 = try Conv2DLayer(f32).init(
    alloc, 1, 6, 5, 1, 0, 1
);
var pool1 = try MaxPool2DLayer(f32).init(
    alloc, 2, 2, 0
);
try model.addLayer(conv1.asLayer());
try model.addLayer(pool1.asLayer());`}
            </CodeBlock>
          </div>
          <div className="col col--6">
            <CodeBlock language="zig">
              {`// Train with extreme control
var trainer = Trainer(f32, .ce).init(
    model, 
    optimizer,
    .{
        .grad_clip_enabled = true,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
    }
);

// Get blazing fast inference
const output = try model.forward(
    input, 
    allocator
);`}
            </CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Zigrad - High performance deep learning framework in Zig with extreme optimization capabilities and systems-level control">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <section className={styles.performanceSection}>
          <div className="container">
            <div className="row">
              <div className="col col--6">
                <Heading as="h2">Extreme Performance</Heading>
                <ul className={styles.performanceList}>
                  <li>2.5x+ speedup over PyTorch on Apple Silicon</li>
                  <li>Tiny binaries under 400KB</li>
                  <li>Zero-overhead abstractions</li>
                  <li>BLAS-accelerated computations</li>
                </ul>
              </div>
              <div className="col col--6">
                <Heading as="h2">Deep Control</Heading>
                <ul className={styles.performanceList}>
                  <li>Fine-grained memory management</li>
                  <li>Flexible performance tradeoffs</li>
                  <li>Hardware-specific optimizations</li>
                  <li>Transparent allocation patterns</li>
                </ul>
              </div>
            </div>
          </div>
        </section>
        <CodeSection></CodeSection>
      </main>
    </Layout>
  );
}
