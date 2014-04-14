package org.lemming.pull_push;

/**
 * A pushable processor is a well that processes objects and then
 * pushes the results to multiple downstream wells.
 * 
 * @param <T1> the input type
 * @param <T2> the output type
 */
public PushableProcessor<T1, T2> implements Well<T2> {
        Processor<T1, T2> processor;
        Array<Well<T1>> downstream;

        PullableProcessor(Processor<T1, T2> processor, Array<Well<T1>> downstream) {
                this.processor = processor;
                this.downstream = downstream;
        }

        @Override
        public void beforeRun() {
                for (Well<T2> well : downstream) {
                        downstream.beforeRun();
                }
        }

        public void afterRun() {
                for (Well<T2> well : downstream) {
                        downstream.afterRun();
                }
        }

        @Override
	public void process(T1 input) {
                T2 result = processor.process(input)
                if (result != null) {
                    for (Well<T2> well : downstream) {
                        well.process(result);
                    }
                }
        }
}
