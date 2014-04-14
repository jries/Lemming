package org.lemming.pull_push;

/**
 * A pullable processor is a source that pulls objects from a
 * upstream Source and processes them with the given processor.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public PullableProcessor<T1, T2> implements Source<T2> {
        Source<T1> upstream;
        Processor<T1, T2> processor;

        PullableProcessor(Source<T1> upstream, Processor<T1, T2> processor) {
            this.upstream = upstream;
            this.processor = processor;
        }

        @Override
        public void beforeRun() {
            upstream.beforeRun();
        }

        public void afterRun() {
            upstream.afterRun();
        }

        @Override
	public boolean hasMoreOutputs() {
            return upstream.hasMoreOutputs();
        }

        @Override
	public T2 newOutput() {
            T1 input = upstream.newOutput();
            if (input == null) {
                return null;
            } else {
                return processor.process(input);
            }
        }
}
