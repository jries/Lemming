package org.lemming.pipeline;

public abstract class SingleRunModule extends AbstractModule {

	@Override
	public void run() {
		
		if (!inputs.isEmpty()){
			beforeRun();
			if (inputs.get(iterator)!=null)
				while (inputs.get(iterator).isEmpty()) pause(10);
			while (running) {
				if (Thread.currentThread().isInterrupted()) break;
				Element data = nextInput();
				process(data);
			}
			afterRun();	
			return;
		}
		if (!outputs.isEmpty()){
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted()) break;
				Element data = (Element) new Object();
				process(data);
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
	
	public abstract void process(Element data);
}
