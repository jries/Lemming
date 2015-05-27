package org.lemming.pipeline;

public abstract class SingleRunModule extends AbstractModule {

	@Override
	public void run() {

		if (!inputs.isEmpty()) {
			if (inputs.keySet().iterator().hasNext())
				iterator = inputs.keySet().iterator().next();
			beforeRun();
			if (inputs.get(iterator) != null) {
				while (inputs.get(iterator).isEmpty())
					pause(10);
				while (running) {
					if (Thread.currentThread().isInterrupted())
						break;
					Element data = nextInput();
					process(data);
				}
			}
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) {
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = process(null);
				newOutput(data);
			}
			afterRun();
			return;
		}
	}

	protected void afterRun() {
	}

	protected void beforeRun() {
	}
	
	/**
	 * Method to be overwritten by children of this class.
	 * @param data - data to process
	 * @return Element
	 */
	public abstract Element process(Element data);
}
