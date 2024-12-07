import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from "./api.module.css";

function APIDocsContent() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load WASM and main.js
    const script = document.createElement('script');
    script.src = '/api-docs/main.js';
    script.async = true;
    script.onload = () => setIsLoading(false);
    // const l = document.createElement("link");
    // l.rel = "stylesheet";
    // l.href = "/api-docs/api-styles.css;"
    // document.head.appendChild(l);
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div className="api-docs-container padding-vert--sm padding-horiz--lg">
      <div id="navWrap">
        {/* <input type="search" id="search" autocomplete="off" spellcheck="false" placeholder="`s` to search, `?` to see more options"> */}
        <input type="search" id="search" className={styles.search} placeholder="`s` to search, `?` to see more options">
        </input>
        <div id="sectNav" className="hidden"><ul id="listNav"></ul></div>
      </div>
      <section>
        <p id="status">Loading...</p>
        <h2 id="hdrName" className="padding-top--md hidden"><span></span><a href="#">[src]</a></h2>
        <div id="fnProto" className="hidden">
          <pre><code id="fnProtoCode"></code></pre>
        </div>
        <div id="tldDocs" className="hidden"></div>
        <div id="sectParams" className="hidden">
          <h3>Parameters</h3>
          <div id="listParams">
          </div>
        </div>
        <div id="sectFnErrors" className="hidden">
          <h3>Errors</h3>
          <div id="fnErrorsAnyError">
            <p><span className="tok-type">anyerror</span> means the error set is known only at runtime.</p>
          </div>
          <div id="tableFnErrors"><dl id="listFnErrors"></dl></div>
        </div>
        <div id="sectSearchResults" className="hidden">
          <h3>Search Results</h3>
          <ul id="listSearchResults"></ul>
        </div>
        <div id="sectSearchNoResults" className="hidden">
          <h3>No Results Found</h3>
          <p>Press escape to exit search and then '?' to see more options.</p>
        </div>
        <div id="sectFields" className="hidden">
          <h3>Fields</h3>
          <div id="listFields">
          </div>
        </div>
        <div id="sectTypes" className="hidden">
          <h3>Types</h3>
          <ul id="listTypes" className="columns">
          </ul>
        </div>
        <div id="sectNamespaces" className="hidden">
          <h3>Namespaces</h3>
          <ul id="listNamespaces" className="columns">
          </ul>
        </div>
        <div id="sectGlobalVars" className="hidden">
          <h3>Global Variables</h3>
          <table>
            <tbody id="listGlobalVars">
            </tbody>
          </table>
        </div>
        <div id="sectValues" className="hidden">
          <h3>Values</h3>
          <table>
            <tbody id="listValues">
            </tbody>
          </table>
        </div>
        <div id="sectFns" className="hidden">
          <h3>Functions</h3>
          <dl id="listFns">
          </dl>
        </div>
        <div id="sectErrSets" className="hidden">
          <h3>Error Sets</h3>
          <ul id="listErrSets" className="columns">
          </ul>
        </div>
        <div id="sectDocTests" className="hidden">
          <h3>Example Usage</h3>
          <pre><code id="docTestsCode"></code></pre>
        </div>
        <div id="sectSource" className="hidden">
          <h3>Source Code</h3>
          <pre><code id="sourceText"></code></pre>
        </div>
      </section>
      <div id="helpDialog" className="hidden">
        <h2>Keyboard Shortcuts</h2>
        <dl><dt><kbd>?</kbd></dt><dd>Show this help dialog</dd></dl>
        <dl><dt><kbd>Esc</kbd></dt><dd>Clear focus; close this dialog</dd></dl>
        <dl><dt><kbd>s</kbd></dt><dd>Focus the search field</dd></dl>
        <dl><dt><kbd>u</kbd></dt><dd>Go to source code</dd></dl>
        <dl><dt><kbd>↑</kbd></dt><dd>Move up in search results</dd></dl>
        <dl><dt><kbd>↓</kbd></dt><dd>Move down in search results</dd></dl>
        <dl><dt><kbd>⏎</kbd></dt><dd>Go to active search result</dd></dl>
      </div>
      <script src="main.js"></script>
    </div>
  );
}

export default function APIPage(): JSX.Element {
  return (
    <Layout
      title="API Documentation"
      description="Zigrad API Documentation">
      <main>
        <BrowserOnly>
          {() => <APIDocsContent />}
        </BrowserOnly>
      </main>
    </Layout>
  );
}
