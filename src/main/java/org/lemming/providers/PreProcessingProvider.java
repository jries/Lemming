package org.lemming.providers;

import org.lemming.factories.PreProcessingFactory;

public class PreProcessingProvider extends AbstractProvider<PreProcessingFactory> {

	public PreProcessingProvider() {
		super(PreProcessingFactory.class);
	}
	
	public static void main(String[] args) {
		final PreProcessingProvider provider = new PreProcessingProvider();
		System.out.println( provider.echo() );
	}

}
